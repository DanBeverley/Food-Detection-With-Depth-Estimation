import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner, human_labeling_interface


class FoodTrainingSystem:
    def __init__(self, config:dict):
        self.config = self._validate_config(config)
        self.device = torch.device(config["device"])

        # Initialize components
        self._init_components()
        self._init_optimization()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def _validate_config(self, config: dict) -> dict:
        """Validate and set default configuration values"""
        defaults = {
            "batch_size": 32,
            "epochs": 100,
            "lr": 3e-4,
            "mixed_precision": True,
            "early_stopping_patience": 10,
            "qat_start_epoch": 5,
            "trt_validation_freq": 10,
        }
        return {**defaults, **config}

    def _init_components(self):
        """Initialize all model components"""
        # Data handling
        self.nutrition_mapper = NutritionMapper(api_key=self.config["usda_key"])
        self.dataset = UECFoodDataset(
            root_dir=self.config["data_root"],
            transform=get_train_transforms(),
            nutrition_mapper=self.nutrition_mapper
        )

        # Models
        self.detector = FoodDetector(
            quantized=self.config["quantize"],
            device=self.device
        )

        self.classifier = FoodClassifier(
            num_classes=256,
            active_learner=ActiveLearner(
                self.dataset,
                self.config["unlabeled_pool"]
            ) if self.config.get("active_learning") else None,
            device=self.device
        )

    def _init_optimization(self):
        """Initialize optimization components"""
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW([
            {
                "params": self.classifier.model.parameters(),
                "lr": self.config["lr"]
            },
            {
                "params": self.detector.model.parameters(),
                "lr": self.config["lr"] * 0.1
            }
        ], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    async def train_epoch(self, epoch):
        """Training for one epoch"""
        self.classifier.model.train()
        self.detector.model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            loss = await self._train_step(images, targets, batch_idx, epoch)
            total_loss += loss

            if batch_idx % 10 == 0:
                self._log_progress(epoch, batch_idx, loss)
        return total_loss/len(self.train_loader)

    async def _train_step(self, images, targets, batch_idx: int, epoch: int) -> float:
        """Single training step"""
        images = images.to(self.device, non_blocking=True)

        # QAT handling
        if self._should_start_qat(epoch):
            self._initialize_qat()

        # Mixed precision training
        with autocast(enabled=self.config["mixed_precision"]):
            loss = self._forward_pass(images, targets)

        # Optimization
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Active learning
        if self.config.get("active_learning") and batch_idx % 100 == 0:
            await self._active_learning_step()

        return loss.item()

    def _forward_pass(self, images, targets) -> torch.Tensor:
        """Forward pass through both models"""
        detections = self.detector.detect(images, return_masks=True)
        class_outputs = self.classifier.model(images)

        return self._calculate_loss(
            class_outputs,
            targets,
            detections
        )

    def _calculate_loss(self, outputs, targets, detections) -> torch.Tensor:
        """Calculate combined loss"""
        weights = self.config.get("loss_weights", {
            "classification": 1.0,
            "nutrition": 0.3,
            "portion": 0.2
        })

        losses = {
            "classification": F.cross_entropy(
                outputs["class"],
                targets["labels"]
            ),
            "nutrition": F.mse_loss(
                outputs["nutrition"],
                targets["nutrition"]
            ),
            "portion": F.l1_loss(
                outputs["portions"],
                targets["portions"]
            )
        }

        return sum(weight * loss for loss, weight in zip(losses.values(), weights.values()))

    def _should_start_qat(self, epoch: int) -> bool:
        """Check if QAT should be initialized"""
        return (
            self.config["qat_enabled"] and
            epoch == self.config["qat_start_epoch"] and
            not hasattr(self, 'qat_initialized')
        )

    def _initialize_qat(self):
        """Initialize Quantization Aware Training"""
        self.detector.prepare_for_qat()
        self.classifier.prepare_for_qat()
        self.detector.calibrate_model(self.val_loader)
        self.quat_initialized = True

    async def run_training(self):
        """Main training loop"""
        await self.initialize()

        for epoch in range(self.config["epochs"]):
            self.current_epoch = epoch

            # Training
            train_loss = await self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Model saving and early stopping
            if self._handle_validation_results(train_loss, val_loss):
                break

            # TensorRT export
            if self._should_export_trt(epoch):
                self._export_trt_checkpoint(epoch)

    def _handle_validation_results(self, train_loss: float, val_loss: float) -> bool:
        """Handle validation results and early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best")
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        self._save_checkpoint(f"epoch_{self.current_epoch}")

        # Early stopping
        if self.early_stopping_counter >= self.config["early_stopping_patience"]:
            print(f"Early stopping triggered after {self.current_epoch + 1} epochs")
            return True

        return False

    def _should_export_trt(self, epoch: int) -> bool:
        """Check if TensorRT export should be performed"""
        return (
                self.device.type == "cuda" and
                epoch % self.config["trt_validation_freq"] == 0
        )

if __name__ == "__main__":
    config = {
        "trt_engine_path": "models/yolov8_food.trt",
        "data_root": "/path/to/uecfood256",
        "unlabeled_pool": "/path/to/unlabeled",
        "usda_key": "your_api_key",
        "active_learning": True,
        "qat_enabled": True,
        "mixed_precision": True,
        "quantize": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_weights": {
            "classification": 1.0,
            "nutrition": 0.3,
            "portion": 0.2
        }
    }

    system = FoodTrainingSystem(config)
    asyncio.run(system.run_training())