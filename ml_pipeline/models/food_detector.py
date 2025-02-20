from typing import Optional, List, Union, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.version import cuda

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
                self.context = self.trt_engine.create_execution_context
                self._allocate_buffers()
            else:
                logging.warning("TensorRT engine not found, using Pytorch model")
        except Exception as e:
            logging.error(f"TensorRT setup failed: {e}")
            self.trt_engine = None
            self.stream = cuda.Stream()
            self.bindings = []
            self.inputs = []
            self.outputs = []
            self.input_shape = (3, 640, 640)

    def build_trt_engine(self, output_path="yolov8.trt") -> None:
        """Proper TensorRT export implementation"""
        from torch2trt import torch2trt
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        model_trt = torch2trt(self.model,
                              [dummy_input],
                              fp16_mode=True,
                              max_workspace_size=1 << 25)
        with open(output_path, "wb") as f:
            f.write(model_trt.engine.serialize())

    def __del__(self) -> None:
        if self.context:
            self.context.__del__()
        if self.trt_engine:
            self.trt_engine.__del__()

    def detect_batch(self, images:Union[str, np.ndarray], batch_size:int=32) -> List[Dict[str, Union[List, float, int]]]:
        """Process images in batches"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self.model(batch, conf=self.conf_threshold)
            results.extend(self._process_results(batch_results, return_masks=False, original_shape=(640,640)))
        return results

    @staticmethod
    def preprocess_image(image:Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image for YOLO model"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError(f"Invalid dtype: {image.dtype}, expected uint8")
            if image.max() > 255 or image.min() < 0:
                raise ValueError("Image values out of [0, 255] range")
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            if image.shape[2] == 3 and image.dtype == np.uint8:
                if image[0,0,0] > image[0,0,2]: # Crude BGR check
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be a path string or numpy array")
        return image

    def build_trt_engine(self, output_path="yolov8.trt"):
        """Export YOLOv8 to TensorRT engine with proper optimization"""
        ModelOptimizer.export_tensorrt(self.model, output_path=output_path)

    def _preprocess(self, image:Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image for TensorRT inference"""
        # Resize and Normalize
        img = cv2.resize(image, self.input_shape[1:][::-1])
        img = img.transpose(2,0,1) # HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32)/255.0
        # Expand dimension if needed
        if len(img.shape)==3:
            img = np.expand_dims(img, axis=0)
        return img

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
        image = self.preprocess_image(image)

        # Run inference
        results = self.model(image, conf = self.conf_threshold,
                             verbose=False, device=self.device)
        return self._process_results(results, return_masks, image.shape[:2])

    @staticmethod
    def _process_results(results:List, return_masks:bool,
                         original_shape:Tuple) -> List[Dict[str, Union[np.ndarray, float, int]]]:
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            # Masks if available
            masks = result.masks if hasattr(result, "masks") else None

            for i, box in enumerate(boxes):
                detection = {"bbox":box,
                             "confidence": float(result.boxes.conf[i]),
                             "class_id":int(result.boxes.cls[i])}
                if return_masks and masks:
                    # Convert mask to original image dimensions
                    mask = masks[i].data.cpu().numpy().squeeze()
                    mask = cv2.resize(mask, original_shape[::-1],
                                      interpolation=cv2.INTER_NEAREST)
                    detection["mask"] = mask.astype(np.float32)
                detections.append(detection)
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
        self.model = ModelOptimizer.quantize_model(self.model)

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

