import torch
import numpy as np
from torch import nn
from torch.ao.quantization import quantize_dynamic
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
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
        self.trt_engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.fp16_mode = True
        self.workspace_size = 1<<30 # Around 1GB
        self.input_shape = (3, 640, 640)

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
            self.model.export(format="torchscript", imgsz=640, optimizer=True)
            self.scripted_model = torch.jit.load("yolov8.torchscript")
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
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            if image.shape[2] == 3 and image.dtype == np.uint8:
                if image[0,0,0] > image[0,0,2]: # Crude BGR check
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be a path string or numpy array")
        return image
    def build_trt_engine(self, output_path="yolov8.trt"):
        """Export YOLOv8 to TensorRT engine with proper optimization"""
        self.model.export(format="engine", imgsz=self.input_shape[1:],
                          half=self.fp16_mode, workspace=self.workspace_size,
                          simplify=True, int8=False, device=0, verbose=True)
        if Path(output_path).exists():
            self._load_engine(output_path)

    def _load_engine(self, engine_path):
        """Load pre-built TensorRT engine"""
        logger = trt.Logger(trt.Logger.WARNING)
        # Read engine file
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        # Create execution context
        self.context = self.trt_engine.create_execution_context()
        # Allocate memory buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate input/output buffers for TensorRT"""
        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding))
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                self.inputs.append({"host":host_mem, "device":device_mem})
            else:
                self.outputs.append({"host":host_mem, "device":device_mem})

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

        if self.scripted_model:
            image_tensor = torch.from_numpy(image).to(self.device)
            if self.device.type == "cuda":
                image_tensor = image_tensor.half() # Convert to FP16
            results = self.scripted_model(image_tensor)
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
            masks = result.masks if hasattr(result, "masks") else None

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
                    detection["mask"] = mask.astype(np.float32)
                detections.append(detection)
        return detections

    def detect_TRT(self, image):
        """Run inference using TensorRT engine"""
        if not self.trt_engine:
            raise RuntimeError("TensorRT engine not initialized!")

        # Preprocess input
        preprocessed = self._preprocess(image)
        np.copyto(self.inputs[0]['host'], preprocessed.ravel())

        # Transfer data to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Transfer predictions back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        # Post-process outputs
        output = self.outputs[0]['host']
        return self._postprocess(output, image.shape)

    def _postprocess(self, output, orig_shape):
        """Convert raw TensorRT output to YOLO results format"""
        # Reshape output
        num_detections = int(output[0])
        boxes = output[1:1 + num_detections * 4].reshape(-1, 4)
        scores = output[1 + num_detections * 4:1 + num_detections * 5]
        class_ids = output[1 + num_detections * 5:1 + num_detections * 6].astype(int)

        # Scale boxes to original image size
        boxes = self._scale_boxes(boxes, orig_shape)

        # Convert to list of detections
        detections = []
        for i in range(num_detections):
            detections.append({
                'bbox': boxes[i],
                'confidence': float(scores[i]),
                'class_id': int(class_ids[i])
            })

        return detections

    def _scale_boxes(self, boxes, orig_shape):
        """Rescale boxes from input size to original image size"""
        input_height, input_width = self.input_shape[1:]
        orig_height, orig_width = orig_shape[:2]

        # Calculate scaling factors
        gain = min(input_height / orig_height, input_width / orig_width)
        pad_x = (input_width - orig_width * gain) / 2
        pad_y = (input_height - orig_height * gain) / 2

        # Scale boxes
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= gain

        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_height)

        return boxes

    def warmup(self, iterations=10):
        """Warmup TensorRT engine for stable performance"""
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(iterations):
            self.detect(dummy_input)

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

    def _postprocess_mask(self, mask, threshold=.5):
        """Apply morphological operations to clean mask"""
        # Binarize mask
        _, binary_mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
        # Morphological closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return processed_mask

    def quantize(self):
        self.model = torch.quantization.quantize_dynamic(self.model.to("cpu"),
                                                         {nn.Conv2d, nn.Linear},
                                                         dtype=torch.qint8)
        self.model.eval()

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

