import torch
import asyncio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner, human_labeling_interface


class FoodTrainingSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        # Initialize components
        self.nutrition_mapper = NutritionMapper(api_key=config["usda_key"])
        self.dataset = UECFoodDataset(root_dir=config["data_root"],
                                      transform=train_transform,
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
                                                    transform=val_transform),
                                     batch_size=self.config["batch_size"],
                                     num_workers=4, collate_fn=collate_fn)

    async def train_epoch(self, epoch):
        self.classifier.model.train()
        self.detector.model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            # Mixed precision training
            with autocast():
                # Detection
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
                      "losses":(train_loss, val_loss)}
        torch.save(checkpoint, f"checkpoints/{timestamp}_epoch{epoch}.pt")

    def _quantize_models(self):
        self.detector.optimize_for_mobile()
        self.classifier.quantize()
        # Export to Tensor RT
        if self.device.type == "cuda":
            self.detector.build_trt_engine()

if __name__ == "__main__":
    config = {
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