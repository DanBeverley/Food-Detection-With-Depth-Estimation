import torch
import numpy as np
from torch.ao.quantization import quantize_dynamic
from ultralytics import YOLO
from pathlib import Path
import logging
import cv2

class FoodDetector:
    def __init__(self, model_path:str=None, confidence:float=0.5, device:torch.device=None,
                 half_precision:bool=True, quantized:bool=None,  **kwargs):
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
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Load pretrained YOLOv8 and fine tune for food detection
                self.model = YOLO("yolov8n.pt")
                logging.info("Usingp pretrained YOLOv8n model")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

        self.model.to(device)
        # Quantization
        if quantized:
            self._fuse_layers()
            self.model = quantize_dynamic(self.model.to("cpu"),{torch.nn.Conv2d,
                                                                torch.nn.Linear},
                                          dtype = torch.qint8)
        # Half precision optimization
        if half_precision and device.type == "cuda":
            self.model = self.model.half()

        # Export to TorchScript
        self.scripted_model = None
        if Path("yolo_scripted.pt").exists():
            self.scripted_model = torch.jit.load("yolo_scripted.pt")

        self._export_torchscript()
    def _fuse_layers(self):
        """Fuse Conv+BN+ReLU layers for quantization compatibility"""
        for module_name, module in self.model.named_children():
            if "features" in module_name:
                torch.quantization.fuse_modules(module, [["0.0","0.1","0.2"]], # Conv2d + BN + ReLU
                                                inplace = True)
    def _export_torchscript(self):
        if not Path("yolov8_scripted.pt").exists():
            dummy_input = torch.randn(1,3,640,640).to(self.device)
            if self.device.type == "cuda":
                dummy_input = dummy_input.half()
            self.scripted_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(self.scripted_model, "yolov8_scripted.pt")
    def optimize_for_mobile(self):
        quantized_model = torch.quantization.quantize_dynamic(self.model,
                                                              {torch.nn.Linear,
                                                               torch.nn.Conv2d},
                                                              dtype=torch.qint8)
        torch.save(quantized_model.state_dict(), "yolo_quantized.pt")

    def preprocess_image(self, image):
        """Preprocess image for YOLO model"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and image.ndim == 2:
            if image.shape[2] == 3 and image.dtype == np.uint8:
                if image[0,0,0] > image[0,0,2]: # Crude BGR check
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be a path string or numpy array")
        return image

    def detect(self, image, return_masks=False):
        """
        Detect food items in image with optional segmentation masks
        Args:
            image: numpy array or path to image
            return_masks: whether to return segmentation masks
        Returns:
            list of dict containing detection results
        """
        if self.scripted_model:
            results = self.scripted_model(image)
        else:
            results = self.model(image)
        image = self.preprocess_image(image)

        # Run inference
        results = self.model(image, conf = self.conf_threshold,
                             verbose=False, device=self.device)
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            # Masks if available
            masks = result.masks if hasattr(result, "mask") else None

            for i, box in enumerate(boxes):
                detection = {"bbox":box,
                             "confidence": float(result.boxes.conf[i]),
                             "class_id":int(result.boxes.cls[i])}
                if return_masks and masks:
                    # Convert mask to original image dimensions
                    mask = masks[i].data.cpu().numpy().squeeze()
                    orig_h, orig_w = image.shape[:2]
                    # Resize mask to match original image size
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    detection["mask"] = mask.astype(bool)
                detections.append(detection)
        return detections
    def _postprocess_mask(self, mask, threshold=.5):
        """Apply morphological operations to clean mask"""
        # Binarize mask
        _, binary_mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
        # Morphological closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return processed_mask

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

