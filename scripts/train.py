import torch.nn.functional as F
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.config_loader import get_params
from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner
from ml_pipeline.models.food_detector import FoodDetector

class FoodTrainingSystem:
    def __init__(self, cfg:dict):
        #self.config = self._validate_config(cfg = cfg)
        self.config_params = get_params()
        if self.config_params is None:
            raise RuntimeError("Configuration could not be loaded")
        self.device = torch.device(self.config_params["system"]["device"])
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
        if self.config_params.get("resume_from"):
            self._load_checkpoint(self.config_params["resume_from"])

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
    def _validate_config(cfg: dict) -> dict:
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
        self.nutrition_mapper = NutritionMapper(api_key=self.config_params["system"]["usda_key"])
        self.dataset = UECFoodDataset(
            root_dir=self.config_params["paths"]["dataset"],
            transform=get_train_transforms(),
            nutrition_mapper=self.nutrition_mapper
        )
        # Create train/val split
        train_size = int(self.config_params["training"]["train_val_split"] * len(self.dataset))
        val_size = len(self.dataset) - train_size
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=generator
        )

        # Initialize dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.config_params["training"]["batch_size"],
                                                        shuffle=True,
                                                        num_workers=self.config_params["training"]["num_workers"],
                                                        persistent_workers=True,
                                                        pin_memory=True if self.device.type == "cuda" else False,
                                                        drop_last=True)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config_params["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config_params["training"]["num_workers"],
            pin_memory=True if self.device.type == "cuda" else False
        )

        # Models
        self.detector = FoodDetector(
            quantized=False,
            device=self.device
        )
        self.classifier= FoodClassifier(
            num_classes=256,
            active_learner=ActiveLearner(
                self.dataset,
                self.config_params["unlabeled_pool"] #TODO: add this to the config
            ) if self.config_params["training"]["active_learning"] else None,
            device=self.device
        )

    def _init_optimization(self):
        """Initialize optimization components"""
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW([
            {
                "params": self.classifier.model.parameters(),
                "lr": self.config_params["training"]["learning_rate"]
            },
            {
                "params": self.detector.model.parameters(),
                "lr": self.config_params["training"]["learning_rate"] * 0.1
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
        is_accumulation_step = (batch_idx + 1) % self.config_params["training"]["accumulation_steps"]

        # Mixed precision training
        with autocast(enabled=self.config_params["training"]["mixed_precision"]):
            loss = self._forward_pass(val_images, val_targets)

            # Scale loss for gradient accumulations
            if self.config_params["training"]["accumulation_steps"]>1:
                loss = loss/self.config_params["training"]["accumulation_steps"]

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        if not is_accumulation_step:
            # Gradient clipping
            if self.config_params["training"]["gradient_clip_val"]>0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm(
                    self._get_trainable_parameters(),
                    self.config_params["training"]["gradient_clip_val"]
                )

            # Optimization

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def _get_trainable_parameters(self):
        return list(self.classifier.model.parameters()) + list(self.detector.model.parameters())
    def _forward_pass(self, val_images, val_targets):
        """Forward pass through both models"""
        with torch.no_grad():
            detections = self.detector.model(val_images)
        outputs = self.classifier.model(val_images)
        required_keys = ["labels", "nutrition", "portions"]
        for key in required_keys:
            if key not in val_targets:
                raise ValueError(f"Missing key '{key}' in val_targets")
        return self._calculate_loss(outputs, val_targets, detections)

    def _calculate_loss(self, outputs, val_targets, detections) -> float:
        """Calculate combined loss"""
        weights = self.config_params["training"]["loss_weights"]
        cls_loss = F.cross_entropy(outputs["class"], val_targets["labels"])
        nutrition_loss = F.mse_loss(outputs["nutrition"][:, :3],
                                    val_targets["nutrition"][:, :3])
        portion_loss = .7 + F.l1_loss(outputs["portion"], val_targets["portions"]) + \
                       .3 + F.mse_loss(outputs["portion"], val_targets["portions"])
        return (
                weights["classification"] * cls_loss +
                weights["nutrition"] * nutrition_loss +
                weights["portion"] * portion_loss
        )

    def _should_start_qat(self, epoch: int) -> bool:
        """Check if QAT should be initialized"""
        return (
            self.config_params["training"]["qat_enabled"] and
            epoch == self.config_params["training"]["qat_start_epoch"] and
            not getattr(self, "qat_initialized", False)
        )

    def _reinitialized_optimization(self):
        """Reinitialize optimizer and gradient scaler with QAT-enabled parameters"""
        self.optimizer = torch.optim.AdamW([{"params": self.classifier.model.parameters(), "lr":
                                             self.config_params["training"]["learning_rate"]},
                                            {"params":self.detector.model.parameters(), "lr":
                                             self.config_params["training"]["learning_rate"]*0.1}], weight_decay=1e-4)
        self.scaler = GradScaler()
    def _initialize_qat(self):
        """Initialize Quantization Aware Training"""
        self.detector.prepare_for_qat()
        self.classifier.prepare_for_qat()
        self.classifier.calibrate_model(self.val_loader)
        self.detector.calibrate_model(self.val_loader)
        self._reinitialized_optimization()
        self.qat_initialized = True
        logging.info(f"Epoch {self.current_epoch}: Enabled Quantization-Aware Training (QAT)")

    def run_training(self):
        """Main training loop"""
        self.initialize()
        try:
            for epoch in range(self.config_params["training"]["epochs"]):
                self.current_epoch = epoch
                # Training
                train_loss = self.train_epoch(epoch)
                # Validation
                val_loss = self.validate()
                # Self-training: Periodically generate pseudo-labels and augment the dataset
                if self.current_epoch % self.config_params["training"]["self.training_freq"]==0:
                    self._perform_self_training()
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                # Model saving and early stopping
                if self._handle_validation_results(train_loss, val_loss):
                    break
                # TensorRT export
#                if self._should_export_trt(epoch):
#                    self._export_trt_checkpoint(epoch)
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.logger.info("Training completed")

    def _perform_self_training(self):
        """Generate pseudo labels and update the dataset"""
        pool_loader = DataLoader(self.classifier.active_learner.unlabeled_pool,
                                 batch_size = self.config_params["training"]["batch_size"], shuffle=False,
                                 num_workers = self.config_params["training"]["num_workers"], pin_memory=True,
                                 collate_fn=self.classifier.collate_with_paths())
        pseudo_labels = self.classifier.pseudo_label_samples(pool_loader, confidence_threshold=0.95)
        logging.info(f"Added {len(pseudo_labels)} pseudo-labeled samples")
        # Update active learner with pseudo labeled data
        self.classifier.active_learner.update_with_pseudo_labels(pseudo_labels)
        # Recreate training DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            self.classifier.active_learner.current_dataset, batch_size=self.config_params["training"]["batch_size"],
            shuffle=True, num_workers=self.config_params["training"]["num_workers"], persistent_workers=True,
            pin_memory=True if self.device.type == "cuda" else False, drop_last=True)

    def validate(self):
        self.classifier.model.eval()
        self.detector.model.eval()
        total_loss = 0.0
        with torch.no_grad(), autocast(enabled=self.config_params["training"]["mixed_precision"]):
            for val_images, val_targets in self.val_loader:
                val_images = val_images.to(self.device)
                with autocast(enabled=self.config_params["training"]["mixed_precision"]):
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
        if self.early_stopping_counter >= self.config_params["training"]["early_stopping_patience"]:
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
                epoch % self.config_params["tensorrt"]["trt_validation_freq"] == 0
        )

    def _export_trt_checkpoint(self, epoch:int):
        from ml_pipeline.utils.optimization import ModelOptimizer
        ModelOptimizer.export_tensorrt(self.detector.model, input_shape=(3, 640, 640),
                                       output_path=f"detector_epoch_{epoch}.trt")
        ModelOptimizer.export_tensorrt(self.classifier.model, input_shape=(3, 256, 256),
                                       output_path=f"classifier_epoch_{epoch}.trt")

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
    system.run_training()