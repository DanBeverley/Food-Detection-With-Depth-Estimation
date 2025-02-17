import torch
import numpy as np
from ml_pipeline.utils.optimization import ModelOptimizer
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO
from pathlib import Path
import logging
import cv2

class FoodDetector:
    def __init__(self, model_path:str=None, confidence:float=0.5, device:torch.device=None,
                 half_precision:bool=True, quantized:bool=False):
        """
       Initialize the food detector with YOLOv8
       Args:
           model_path: Path to custom trained YOLO model, if None uses pretrained
           confidence: Detection confidence threshold
           device: Device to run model on ('cuda', 'cpu', etc)
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

    def _setup_tensorrt(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.trt_engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.input_shape = (3, 640, 640)
    def __del__(self):
        if self.context:
            self.context.__del__()
        if self.trt_engine:
            self.trt_engine.__del__()

    def detect_batch(self, images:torch.Tensor, batch_size:int=32):
        """Process images in batches"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self.model(batch, conf=self.conf_threshold)
            results.extend(self._process_results(batch_results))
        return results

    def preprocess_image(self, image:torch.Tensor):
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

    def _preprocess(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize and Normalize
        img = cv2.resize(image, self.input_shape[1:][::-1])
        img = img.transpose(2,0,1) # HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32)/255.0
        # Expand dimension if needed
        if len(img.shape)==3:
            img = np.expand_dims(img, axis=0)
        return img

    def detect(self, image, return_masks=False):
        """
        Detect food items in image with optional segmentation masks
        Args:
            image: numpy array or path to image
            return_masks: whether to return segmentation masks
        Returns:
            list of dict containing detection results
        """
        image = self.preprocess_image(image)

        # Run inference
        results = self.model(image, conf = self.conf_threshold,
                             verbose=False, device=self.device)
        return self._process_results(results, return_masks, image.shape[:2])

    def _process_results(self, results, return_masks, original_shape):
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


    def prepare_for_qat(self):
        """Modify model for quantization-aware training"""
        from pytorch_quantization import quant_modules
        quant_modules.initialize()

        # Replace layers with quantized versions
        self.model = quant_modules.quantize_model(self.model)

    def calibrate_model(self, calib_loader):
        """Run calibration for INT8 quantization"""
        self.model.eval()
        with torch.no_grad():
            for images, _ in calib_loader:
                self.model(images.to(self.device))

    def quantize(self):
        self.model = ModelOptimizer.quantize_model(self.model)

    def train(self, data_yaml:str, epochs:int = 100, batch_size:int=16,
              image_size:int=640):
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

