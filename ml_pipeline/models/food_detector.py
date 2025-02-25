from typing import Optional, List, Union, Dict, Tuple

import torch
import numpy as np
from tensorrt_bindings import ExecutionContextAllocationStrategy
from torch.utils.data import DataLoader
import pycuda.driver as cuda
import pycuda.autoinit

from ml_pipeline.utils.optimization import ModelOptimizer
import tensorrt as trt
from ultralytics import YOLO
from pathlib import Path
import logging
import cv2

class FoodDetector:
    def __init__(self, model_path:Optional[str]=None, confidence:float=0.5,
                 device:Optional[torch.device]=None,
                 half_precision:bool=True, quantized:bool=False):
        """
       Initialize the food detector with YOLOv8
       Args:
           model_path: Path to custom trained YOLO model, if None uses pretrained
           confidence: Detection confidence threshold
           device: Device to run model on ('cuda', 'cpu', etc.)
       """
        self.conf_threshold = confidence
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_trt = False # Flag to indicate TensorRT usage
        try:
            self.model = YOLO(model_path if model_path and Path(model_path).exists() else "yolov8n.pt")
            self.model.to(self.device)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
        # Quantization
        if quantized:
            self.quantize()
        # Half precision optimization
        if half_precision and torch.cuda.is_available() and device.type == "cuda":
            self.model = self.model.half()
        self._setup_tensorrt()
    def _load_trt_engine(self) -> trt.ICudaEngine:
        """Load pre-built TensorRT engine"""
        with open("yolov8.trt", "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def _allocate_buffers(self) -> None:
        """Allocate input/output buffers for TensorRT"""
        self.bindings = []
        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) * self.trt_engine.max_batch_size
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
    def _setup_tensorrt(self) -> None:
        try:
            if Path("yolov8.trt").exists():
                self.trt_logger = trt.Logger(trt.Logger.WARNING)
                self.trt_engine = self._load_trt_engine()
                self.context = self.trt_engine.create_execution_context(strategy=ExecutionContextAllocationStrategy)
                self.input_shape = self.trt_engine.get_binding_shape(0)
                self._allocate_buffers()
                logging.info("TensorRT engine loaded successfully")
            else:
                logging.warning("TensorRT engine not found, using Pytorch model")
                self.build_trt_engine()
        except Exception as e:
            logging.error(f"TensorRT setup failed: {e}")
            self.trt_engine = None
            self.stream = cuda.Stream()
            self.bindings = []
            self.input_shape = (3, 640, 640)
            self.context = None
            self.inputs = []
            self.outputs = []

    def __del__(self) -> None:
        """Safe cleaning of CUDA resources"""
        if self.trt_engine:
            del self.trt_engine
        if self.context:
            del self.context
        for buf in self.inputs + self.outputs:
            if self.device in buf:
                cuda.memfree(buf["device"])

    def detect_batch(self, images:Union[str, np.ndarray], batch_size:int=32) -> List[Dict[str, Union[List, float, int]]]:
        """Process images in batches"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self.model(batch, conf=self.conf_threshold, verbose=False)
            for result, img in zip(batch_results, batch):
                original_shape = img.shape[:2] if isinstance(img, np.ndarray) else cv2.imread(img).shape[:2]
                processed = self._process_results(result, return_masks=False, original_shape=original_shape)
                results.extend(processed)
        return results

    @staticmethod
    def preprocess_image(image:Union[str, np.ndarray], color_format:str="RGB") -> np.ndarray:
        """Preprocess image for YOLO model"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError(f"Invalid dtype: {image.dtype}, expected uint8")
            if image.max() > 255 or image.min() < 0:
                raise ValueError("Image values out of [0, 255] range")
            if image.ndim == 3 and image.shape[2] == 3 and color_format == "BGR":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be a path string or numpy array")
        return image

    def build_trt_engine(self, output_path="yolov8.trt"):
        """Export YOLOv8 to TensorRT engine with proper optimization"""
        #Export YOLO model ONNX first
        onnx_path = "yolov8.onnx"
        self.model.export(format="onnx", dynamic=True)
        ModelOptimizer.export_tensorrt(onnx_path, output_path=output_path)

    def _preprocess(self, image:Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image for TensorRT inference"""
        # Resize and Normalize
        h, w = image.shape[:2]
        scale = min(self.input_shape[1]/w, self.input_shape[2]/h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(image, new_size)
        img = np.pad(img, ((0, self.input_shape[1]-new_size[1]),
                           (0, self.input_shape[2] - new_size[0]), (0,0)))
        return img.transpose(2,0,1).astype(np.float32)/255.0

    def detect(self, image:Union[str, np.ndarray],
               return_masks:bool=False) -> List[Dict[str, Union[List, float, int]]]:
        """
        Detect food items in image with optional segmentation masks
        Args:
            image: numpy array or path to image
            return_masks: whether to return segmentation masks
        Returns:
            list of dict containing detection results
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1,2,0)
        if self.use_trt:
            return self._infer_trt(image, return_masks)
        else:
            image = self.preprocess_image(image)
            # Run inference
            results = self.model(image, conf = self.conf_threshold,
                                verbose=False, device=self.device)
            single_result = results[0]
            return self._process_results(single_result, return_masks, image.shape[:2])

    def _infer_trt(self, image:Union[str, np.ndarray], return_masks:bool) -> List[Dict[str, Union[np.ndarray, float, int]]]:
        # Placeholder for TensorRT inference
        preprocessed = self._preprocess(image)
        # Copy preprocessed image to device memory
        np.copyto(self.inputs[0]["host"], preprocessed.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle = self.stream.handle)
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
        self.stream.synchronize()
        return self._process_results(self.outputs[0]["host"],
                                     return_masks, image.shape[:2])
    @staticmethod
    def _process_results(result, return_masks:bool,
                         original_shape:Tuple) -> List[Dict[str, Union[np.ndarray, float, int]]]:
        # Process results
        detections = []
        # YOLOv8 returns a single Results object for one image
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        masks = result.masks if return_masks and hasattr(result, "masks") else None

        for i in range(boxes.shape[0]):
            box = boxes[i]
            detection = {
                "bbox": tuple(box),
                "confidence": float(confs[i]),
                "class_id": int(classes[i])
            }
            if masks and return_masks:
                mask = masks[i].data.cpu().numpy().squeeze()
                # original_shape is (height, width), cv2.resize expects (width, height)
                mask = cv2.resize(mask, original_shape[::-1], interpolation=cv2.INTER_NEAREST)
                detection["mask"] = mask.astype(np.float32)
            detections.append(detection)
        if not detections:
            logging.warning("No detections were found...")
        return detections

    @staticmethod
    def prepare_for_qat():
        """Modify model for quantization-aware training"""
        from pytorch_quantization import quant_modules
        quant_modules.initialize()

    def calibrate_model(self, calib_loader:DataLoader) -> None:
        """Run calibration for INT8 quantization"""
        self.model.eval()
        with torch.no_grad():
            for images, _ in calib_loader:
                self.model(images.to(self.device))

    def quantize(self) -> None:
        """
            Quantize the model using ModelOptimizer.
            Note: Requires ModelOptimizer to be implemented.
            """
        try:
            self.model = ModelOptimizer.quantize_model(self.model)
        except NotImplementedError:
            logging.warning("ModelOptimizer not implemented. Quantization skipped.")

    def train(self, data_yaml:str, epochs:int = 100, batch_size:int=16,
              image_size:int=640) -> None:
        """
        Train/Finetune YOLO model on custom dataset
        Args:
            data_yaml: Path to data configuration file
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
        """
        try:
            self.model.train(data=data_yaml,
                             epochs=epochs,
                             batch_size=batch_size,
                             imgsz=image_size,
                             device=self.device)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise

