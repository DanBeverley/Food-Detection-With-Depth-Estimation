import asyncio

import torch.nn.functional as F
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner, human_labeling_interface
from ml_pipeline.models.food_detector import FoodDetector


class FoodTrainingSystem:
    def __init__(self, cfg:dict):
        self.config = self._validate_config(cfg = cfg)
        self.device = torch.device(cfg["device"])

        self._setup_logging()

        # Initialize components
        self._init_components()
        self._init_optimization()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def initialize(self):
        """Final initialization before training"""
        # Move models to device
        self.classifier.model.to(self.device)
        self.detector.model.to(self.device)

        # Load checkpoints if resuming
        if self.config.get("resume_from"):
            self._load_checkpoint(self.config["resume_from"])

    def _setup_logging(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _validate_config(self, cfg: dict) -> dict:
        """Validate and set default configuration values"""
        required_keys = ["data_root", "usda_key"]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required configuration key: {key}")
        defaults = {
            "batch_size": 32,
            "epochs": 100,
            "lr": 3e-4,
            "mixed_precision": True,
            "early_stopping_patience": 10,
            "qat_enabled":False,
            "qat_start_epoch": 5,
            "trt_validation_freq": 10,
            "gradient_clip_val":1.0,
            "accumulation_steps":1,
            "num_workers":os.cpu_count(),
            "train_val_split":0.8
        }
        return {**defaults, **cfg}

    def _init_components(self):
        """Initialize all model components"""
        # Data handling
        self.nutrition_mapper = NutritionMapper(api_key=self.config["usda_key"])
        self.dataset = UECFoodDataset(
            root_dir=self.config["data_root"],
            transform=get_train_transforms(),
            nutrition_mapper=self.nutrition_mapper
        )
        # Create train/val split
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        # Initialize dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.config["batch_size"],
                                                        shuffle=True,
                                                        num_workers=self.config.get("num_workers",4),
                                                        persistent_workers=True,
                                                        pin_memory=True if self.device.type == "cuda" else False,
                                                        drop_last=True)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True if self.device.type == "cuda" else False
        )

        # Models
        self.detector = FoodDetector(
            quantized=self.config["quantize"],
            device=self.device
        )
        self.classifier= FoodClassifier(
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

    def train_epoch(self, epoch:int)->float:
        """Training for one epoch"""
        self.classifier.model.train()
        self.detector.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (val_images, val_targets) in enumerate(progress_bar):
            loss = self._train_step(val_images, val_targets, batch_idx,  epoch)
            total_loss += loss

            progress_bar.set_postfix({"loss":f"{loss:.4f}"})
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch: {self.current_epoch} | "
                                 f"Batch: [{batch_idx}/{num_batches}] | "
                                 f"Loss: {loss:.4f} |"
                                 f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return total_loss/num_batches

    def _train_step(self, val_images, val_targets, batch_idx: int, epoch: int) -> float:
        """Single training step"""
        val_images = val_images.to(self.device, non_blocking=True)

        # QAT handling
        if self._should_start_qat(epoch):
            self._initialize_qat()

        # Accumulation steps for larger effective batch size
        is_accumulation_step = (batch_idx + 1) % self.config["accumulation_steps"]

        # Mixed precision training
        with autocast(enabled=self.config["mixed_precision"]):
            loss = self._forward_pass(val_images, val_targets)

            # Scale loss for gradient accumulations
            if self.config["accumulation_steps"]>1:
                loss = loss/self.config["accumulation_steps"]

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        if not is_accumulation_step:
            # Gradient clipping
            if self.config["gradient_clip_val"]>0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm(
                    self._get_trainable_parameters(),
                    self.config["gradient_clip_val"]
                )

            # Optimization

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        # Active learning
        if self.config.get("active_learning") and batch_idx % 100 == 0:
            self._active_learning_step()

        return loss.item()

    def _get_trainable_parameters(self):
        return list(self.classifier.model.parameters()) + list(self.detector.model.parameters())
    def _forward_pass(self, val_images, val_targets):
        """Forward pass through both models"""
        detections = self.detector.detect(val_images, return_masks=True)
        class_outputs = self.classifier.model(val_images)

        return self._calculate_loss(
            class_outputs,
            val_targets,
            detections
        )

    def _calculate_loss(self, outputs, val_targets, detections) -> float:
        """Calculate combined loss"""
        weights = self.config.get("loss_weights", {
            "classification": 1.0,
            "nutrition": 0.3,
            "portion": 0.2
        })

        losses = {
            "classification": F.cross_entropy(
                outputs["class"],
                val_targets["labels"]
            ),
            "nutrition": F.mse_loss(
                outputs["nutrition"],
                val_targets["nutrition"]
            ),
            "portion": F.l1_loss(
                outputs["portions"],
                val_targets["portions"]
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

    def run_training(self):
        """Main training loop"""
        self.initialize()
        try:
            for epoch in range(self.config["epochs"]):
                self.current_epoch = epoch

                # Training
                train_loss = self.train_epoch(epoch)

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
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.logger.info("Training completed")

    def validate(self):
        self.classifier.model.eval()
        self.detector.model.eval()
        total_loss = 0.0
        with torch.no_grad(), autocast(enabled=self.config["mixed_precision"]):
            for val_images, val_targets in self.val_loader:
                val_images = val_images.to(self.device)
                with autocast(enabled=self.config["mixed_precision"]):
                    loss_tensor = self._forward_pass(val_images, val_targets)
                if isinstance(loss_tensor, torch.Tensor):
                    total_loss += loss_tensor.item()
                else:
                    raise TypeError(f"Unexpected loss type: {type(loss_tensor)}")
        return total_loss / len(self.val_loader)

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

    def _save_checkpoint(self, name:str):
        """Save model state"""
        checkpoint = {"epoch": self.current_epoch,
        "classifier": self.classifier.model.state_dict(),
        "detector": self.detector.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "best_val_loss": self.best_val_loss}
        torch.save(checkpoint, f"{name}_checkpoint.pth")
        self.logger.info(f"Saved checkpoint: {name}_checkpoint.pth")

    def _load_checkpoint(self, path:str):
        """Load training state from checkpoint"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint {path} not found")
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.model.load_state_dict(checkpoint["classifier"])
        self.detector.model.load_state_dict(checkpoint["detector"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint["epoch"]

    def _should_export_trt(self, epoch: int) -> bool:
        """Check if TensorRT export should be performed"""
        return (
                self.device.type == "cuda" and
                epoch % self.config["trt_validation_freq"] == 0
        )

    def _export_trt_checkpoint(self, epoch:int):
        from ml_pipeline.utils.optimization import ModelOptimizer
        ModelOptimizer.export_tensorrt(self.detector.model, input_shape=(3, 640, 640),
                                       output_path=f"detector_epoch_{epoch}.trt")
        ModelOptimizer.export_tensorrt(self.classifier.model, input_shape=(3, 256, 256),
                                       output_path=f"classifier_epoch_{epoch}.trt")

    def _active_learning_step(self):
        pool_loader = DataLoader(self.classifier.active_learner.unlabeled_pool, batch_size=32,
                                 num_workers = self.config["num_workers"], shuffle=True,
                                 pin_memory=True)
        samples = self.classifier.get_uncertain_samples(pool_loader)
        labeled_data = human_labeling_interface(samples)
        self.classifier.active_learner.update_dataset(labeled_data)

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
    if torch.cuda.is_available():
        asyncio.run(system.run_training())
    else:
        system.run_training()