from typing import Optional, List, Union, Dict, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.ao.quantization import quantize_dynamic
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
                 device:Optional[Union[str,torch.device]]=None,
                 half_precision:bool=True, quantized:bool=False):
        """
       Initialize the food detector with YOLOv8
       Args:
           model_path: Path to custom trained YOLO model, if None uses pretrained
           confidence: Detection confidence threshold
           device: Device to run model on ('cuda', 'cpu', etc.)
       """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream() if torch.cuda.is_available() else None
        self.bindings = []
        self.input_shape = (3, 640, 640)
        self.trt_engine = None
        self.context = None
        self.conf_threshold = confidence
        self.use_trt = False # Flag to indicate TensorRT usage

        try:
            self.model = YOLO(model_path if model_path and Path(model_path).exists() else "yolov8n.pt").to(self.device)
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
                self.context = self.trt_engine.create_execution_context()
                self.input_shape = self.trt_engine.get_binding_shape(0)
                self._allocate_buffers()
                self.use_trt = True
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

    def close(self) -> None:
        """Safe cleaning of CUDA resources"""
        if self.trt_engine:
            del self.trt_engine
        if self.context:
            del self.context
        for buf in self.inputs + self.outputs:
            if self.device in buf:
                cuda.memfree(buf["device"])

    def detect_batch(self, images:Union[str, np.ndarray], batch_size:int=32) -> list[list[dict[str, list | float | int]] | list[Any]]:
        """
        Process images in batches

        Args:
        images: List of image paths or numpy arrays
        batch_size: Batch size for processing

        Returns:
        List of detection results for all images
        """
        if not isinstance(images, (list, tuple, np.ndarray)):
            raise ValueError("Images must be a list, tuple or numpy array")

        results = []
        total_images = len(images)
        num_batches = (total_images + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch = images[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_results = []
            for img in batch:
                try:
                    if self.use_trt:
                        detections = self._infer_trt(img, return_masks = True)
                    else:
                        detections = self.detect(img, return_masks=False)
                    if not detections:
                        logging.warning("No detections found for image: %s", img)
                    batch_results.append(detections)
                except Exception as e:
                    logging.error("Error processing image: %s. Error: %s", img, e)
                    batch_results.append({})
            results.extend(batch_results)
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
        try:
            onnx_path = "yolov8.onnx"
            self.model.export(format="onnx", dynamic=True, save_half = torch.cuda.is_available())
            ModelOptimizer.export_onnx(self.model, (3,640,640), onnx_path)
            ModelOptimizer.export_tensorrt(onnx_path,(3,640,640), output_path=output_path)
            logging.info(f"TensorRT engine built and saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to build TensorRT engine: {e}")
            raise

    def _preprocess(self, image:Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image for TensorRT inference"""
        # Resize and Normalize
        h, w = image.shape[:2]
        r = min(self.input_shape[1]/w, self.input_shape[2]/h)
        new_w, new_h = int(w*r), int(h*r)
        image_resized = cv2.resize(image, (new_w, new_h))
        image_padded = np.zeros((self.input_shape[1], self.input_shape[2], 3), dtype=image.dtype)
        image_padded[:new_h, :new_w, :] = image_resized
        image_padded = image_padded.transpose((2,0,1)).astype(np.float32)/255.0
        return image_padded

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
        if not self.context or not self.trt_engine:
            raise RuntimeError("TensorRT not properly initialized")
        preprocessed = self._preprocess(image)
        # Copy preprocessed image to device memory
        np.copyto(self.inputs[0]["host"], preprocessed.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle = self.stream.handle)
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
        self.stream.synchronize()
        raw_output = self.outputs[0]["host"]
        return self._parse_trt_output(raw_output, return_masks, image.shape[:2])

    def _parse_trt_output(self, raw_outputs:np.ndarray, return_masks:bool, original_shape:Tuple)->List[Dict[str, Union[np.ndarray, float, int]]]:
        """
        Parse raw TensorRT output into detection dictionaries.

        Args:
            raw_outputs: Raw output from TensorRT engine (e.g., [num_detections, 7]).
            return_masks: Whether to include masks (not supported here).
            original_shape: Original image shape (height, width).

        Returns:
            List of detection dictionaries with keys 'bbox', 'confidence', and 'class_id'.
        """
        try:
            detections = []
            detection_tensor = None
            mask_tensor = None

            for i, binding in enumerate(self.trt_engine):
                shape = self.trt_engine.get_binding_shape(binding)
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
                if "detections" in binding.lower() and "output" not in binding.lower():
                    detection_tensor = raw_outputs[i].reshape(shape).astype(dtype)
                elif "masks" in binding.lower():
                    mask_tensor = raw_outputs[i].reshape(shape).astype(dtype)

                # Process detection tensor
            if detection_tensor is not None:
                for det in detection_tensor:
                    # Assuming detection format: class_id, confidence, x1, y1, x2, y2
                    if det.size >= 6:
                        class_id = int(det[0])
                        confidence = det[1]
                        x1 = det[2] * original_shape[1]
                        y1 = det[3] * original_shape[0]
                        x2 = det[4] * original_shape[1]
                        y2 = det[5] * original_shape[0]

                        if confidence >= self.conf_threshold:
                            detection = {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "class_id": class_id
                            }

                            # Add mask if available
                            if return_masks and mask_tensor is not None:
                                # Assuming mask_tensor has shape (h, w)
                                mask = mask_tensor.reshape(original_shape[0], original_shape[1])
                                detection["mask"] = mask.astype(np.float32)

                            detections.append(detection)
            return detections
        except Exception as e:
            logging.error(f"Failed to parse TensorRT output: {e}")
            raise


    def _process_results(self, result, return_masks:bool,
                         original_shape:Tuple) -> List[Dict[str, Union[np.ndarray, float, int]]]:
        """
        Process results from YOLO model into detection dictionaries.

        Args:
            result: YOLO result object.
            return_masks (bool): Include segmentation masks.
            original_shape (tuple): Original image dimensions.

        Returns:
            list: List of detection dictionaries.
        """
        detections = []
        if result is None or result.is_empty:
            return detections
        # YOLOv8 returns a single Results object for one image
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
#        masks = result.masks if return_masks and hasattr(result, "masks") else None
        masks = result.masks if return_masks and not result.is_only_xyxy else None
        for i in range(len(boxes.shape[0])):
            box = boxes[i].tolist()
            confidence = confs[i].item()
            class_id = int(classes[i].item())
            if confidence < self.conf_threshold:
                continue

            # Normalize coordinates to original image size
            h, w = original_shape
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": confidence,
                "class_id": class_id
            }
            if masks is not None and i < len(masks):
                mask = masks[i].data.cpu().numpy().squeeze()
                # original_shape is (height, width), cv2.resize expects (width, height)
                mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
                detection["mask"] = mask.astype(np.float32)
            detections.append(detection)
        if not detections:
            logging.warning("No detections were found...")
        return detections

    def prepare_for_qat(self):
        """Modify model for quantization-aware training"""
        from pytorch_quantization import quant_modules
        quant_modules.initialize()
        self.model.qconfig = torch.quantization.get_default_qconfig("x86")
        torch.quantization.prepare_qat(self.model, inplace=True)

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
        except NotImplementedError as e:
            logging.error(f"Quantization failed using ModelOptimizer: {e}")
            logging.info("Falling back to PyTorch dynamic quantization...")
            try:
                self.model = quantize_dynamic(
                    self.model.to('cpu'),
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            except Exception as e:
                logging.error(f"Dynamic quantization failed: {e}")
                raise

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

