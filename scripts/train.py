import torch
import asyncio
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.data_processing.dataset_loader import *
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier, ActiveLearner


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
        pass
    async def _active_learning_step(self):
        pass
