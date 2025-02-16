import torch
import asyncio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from ml_pipeline.utils.transforms import get_train_transforms, get_val_transforms
from ml_pipeline.utils.optimization import ModelOptimizer
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner, human_labeling_interface


class FoodTrainingSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.trt_engine_path = config.get("trt_engine_path", "models/yolov8.trt")
        # Initialize components
        self.nutrition_mapper = NutritionMapper(api_key=config["usda_key"])
        self.dataset = UECFoodDataset(root_dir=config["data_root"],
                                      transform=get_train_transforms(),
                                      nutrition_mapper=self.nutrition_mapper)
        self.detector = FoodDetector(quantized=config["quantize"])
        self.classifier = FoodClassifier(num_classes=256
                                         , active_learner=ActiveLearner(self.dataset,
                                                                        config["unlabeled_pool"]),
                                         device=self.device)
        # Optimization tools
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW([{"params":self.classifier.model.parameters()},
                                            {"params":self.detector.model.parameters(),
        "lr":config["lr"]*0.1}], lr=config["lr"], weight_decay=1e-4)

    async def initialize(self):
        "Async initialization"
        await self.dataset.initialize()
        self._prepare_data_loaders()

    def _prepare_data_loaders(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.config["batch_size"],
                                       shuffle=True, num_workers=8, pin_memory=True,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(UECFoodDataset(root_dir=self.config["data_root"],
                                                    transform=get_val_transforms()),
                                     batch_size=self.config["batch_size"],
                                     num_workers=4, collate_fn=collate_fn)

    async def train_epoch(self, epoch):
        self.classifier.model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            # Mixed precision training
            if self.config["qat_enabled"] and epoch == self.config["qat_start_epoch"]:
                self.detector.prepare_for_qat()
                self.detector.calibrate_model(calib_loader)
            with autocast(enabled = self.config["mixed_precision"]):
                detections = self.detector.detect(images, return_masks=True)
                # Classification and Nutrition
                class_outputs = self.classifier.model(images)
                # Multi-task loss
                loss = self._calculate_loss(class_outputs, targets)
            # Backprop
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Active learning every 100 batches
            if batch_idx % 100 == 0:
                await self._active_learning_step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        return total_loss/len(self.train_loader)

    def _calculate_loss(self, outputs, targets):
        # Classification loss
        cls_loss = F.cross_entropy(outputs["class"], targets["labels"])
        # Nutrition estimation loss
        nut_loss = F.mse_loss(outputs["nutrition"], targets["nutrition"])
        # Portion estimation loss
        portion_loss = F.l1_loss(outputs["portions"], targets["portions"])

        return cls_loss+0.3*nut_loss+0.2*portion_loss

    async def _active_learning_step(self):
        uncertain_samples = self.classifier.get_uncertain_samples(self.train_loader)
        labeled_data = await asyncio.to_thread(human_labeling_interface, uncertain_samples)
        self.classifier.active_learner.update_dataset(labeled_data)
        self._prepare_data_loaders()  # Refresh with new data

    async def run_training(self):
        await self.initialize()
        for epoch in range(self.config["epochs"]):
            train_loss = await self.train_epoch(epoch)
            val_loss = self.validate()
            # Save checkpoint
            self._save_checkpoint(epoch, train_loss, val_loss)
            print(f"Epoch {epoch} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            # Quantize after 5 epochs
            if epoch == 5 and self.config["quantize"]:
                self._quantize_models()
            if epoch%10 == 0:
                self._export_trt_checkpoint(epoch)

    def validate(self):
        self.classifier.model.eval()
        total_loss=0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                outputs = self.classifier.model(images)
                loss = self._calculate_loss(outputs, targets)
                total_loss += loss.item()
        return total_loss/len(self.val_loader)

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {"classifier":self.classifier.model.state_dict(),
                      "detector":self.detector.model.state_dict(),
                      "optimizer":self.optimizer.state_dict(),
                      "epoch":epoch,
                      "losses":(train_loss, val_loss),
                      "trt_compatibility": "fp16" if self.config["fp16"] else "fp32",
                      "input_shape": self.detector.input_shape,
                      "calibration_data": self.config["trt_calibration_data"]
                      }
        torch.save(checkpoint, f"checkpoints/{timestamp}_epoch{epoch}.pt")

    def _quantize_models(self):
        ModelOptimizer.quantize_model(self.detector.model)
        ModelOptimizer.quantize_model(self.classifier.model)
        # Export to Tensor RT
        if self.device.type == "cuda":
            self.detector.build_trt_engine()

    def _export_trt_checkpoint(self, epoch):
        """Export best model to TensorRT format"""
        # Load best weights
        self.detector.model = YOLO(f"checkpoints/best_epoch{epoch}.pt")

        # Export to TensorRT
        self.detector.build_trt_engine(output_path=self.trt_engine_path)

        # Validate TRT performance
        trt_metrics = self._validate_trt_performance()
        print(f"TRT Validation: mAP={trt_metrics[0]:.3f}, Latency={trt_metrics[1]:.3f}ms")

    def _validate_trt_performance(self):
        """Benchmark TensorRT model"""
        # Warmup
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.detector.warmup()

        # Latency test
        start_time = time.time()
        for _ in range(100):
            _ = self.detector.detect(dummy_image)
        latency = (time.time() - start_time) * 10  # ms per inference

        # Accuracy test
        val_dataset = UECFoodDataset(transform=get_val_transforms())
        acc = self._evaluate_trt_accuracy(val_dataset)

        return acc, latency

    def _evaluate_trt_accuracy(self, dataset):
        """Validate TensorRT model accuracy"""
        correct = 0
        total = 0

        for img, target in DataLoader(dataset, batch_size=32):
            detections = self.detector.detect(img.numpy())
            # Compare with ground truth
            correct += calculate_accuracy(detections, target)
            total += len(target)

        return correct / total

if __name__ == "__main__":
    config = {
        "trt_engine_path": "models/yolov8_food.trt",
        "trt_validation_freq": 10,  # Validate TRT every 10 epochs
        "trt_calibration_data": "calibration_images/",
        "qat_enabled": True,  # Quantization-Aware Training
        "data_root": "/path/to/uecfood256",
        "unlabeled_pool": "/path/to/unlabeled",
        "usda_key": "your_api_key",
        "batch_size": 64,
        "epochs": 100,
        "lr": 3e-4,
        "quantize": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    system = FoodTrainingSystem(config)
    asyncio.run(system.run_training())