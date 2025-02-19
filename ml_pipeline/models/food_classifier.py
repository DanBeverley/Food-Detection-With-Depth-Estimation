from pathlib import Path
from typing import Any, Union, Iterable, Tuple, List, Optional, Dict, Callable

import cv2

from PIL import Image
import numpy as np

from torch import nn, Tensor
import torch.nn.functional as F
import torch.cuda
from ml_pipeline.data_processing.dataset_loader import UECFoodDataset
from ml_pipeline.utils.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from torch.quantization import quantize_dynamic
from torchvision.datasets import ImageFolder
from multitask_foodnet import MultiTaskFoodNet
import matplotlib.pyplot as plt

class ActiveLearner:
    def __init__(self, base_dataset:UECFoodDataset, unlabeled_pool:Union[str, Path]) -> None :
        self.base_dataset = base_dataset
        self.unlabeled_pool = ImageFolder(unlabeled_pool, transform=base_dataset.transform)
        self.labeled_food = set()
    def update_dataset(self, new_samples:Iterable[Any]) -> None:
        self.labeled_food.update(new_samples)
        self.current_dataset = ConcatDataset([self.base_dataset,
                                              Subset(self.unlabeled_pool, list(self.labeled_food))])

def display_image(img:Union[np.ndarray, Image.Image]) -> None:
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def human_labeling_interface(samples: Iterable[Tuple[Union[np.ndarray, Image.Image], Any]])\
        -> List[Tuple[Union[np.ndarray, Image.Image], str]]:
    labeled = []
    for img, pred in samples:
        display_image(img)
        true_label = input(f"Model Predicted {pred}. Enter correct label: ")
        labeled.append((img, true_label))
    return labeled


class FoodClassifier:
    def __init__(self, model_path:Optional[str]=None, num_classes:int=256,
                 device:Optional[torch.device] = None, quantized:bool = False,
                 active_learner:Optional[ActiveLearner] = None, label_smoothing:float=0.1,
                 calories_scale:int = 500, protein_scale:int = 100) -> None:
        """
        Initialize food classifier with MobileNetV3
        Args:
            model_path: Path to trained model weights
            num_classes: Number of food classes
            device: Device to run model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=="cuda"))

        self.num_classes = num_classes

        # Initialize model
        self.model = MultiTaskFoodNet(num_classes=num_classes,
                                      calories_scale=calories_scale,
                                      protein_scale=protein_scale)

        # Load train weights if provided
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path,
                                                  map_location=self.device))
        # Quantization
        if quantized:
            self.quantize()
        self.active_learner = active_learner
        self.unlabeled_pool = []
        self.label_smoothing = label_smoothing
        self.model = self.model.to(self.device)

        # Setup image preprocessing
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std = [0.229, 0.224, 0.225])])

    def preprocess_image(self, image:Union[str, Path, np.ndarray, Image.Image]) -> Tensor:
        """Preprocess image for classification"""
        # Handle different input types
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        if image is None:
            raise ValueError(f"Could not read image from {image}")
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Validate numpy array input
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim != 3 or image.shape[2] != 3:
                 raise ValueError(f"Invalid image shape: {image.shape}"
                                  "Expected (H, W, 3) RGB array")
            if image.dtype != np.uint8:
                 raise ValueError(f"Invalid dtype: {image.shape}"
                                  "Expected uint8 (0-255 range)")
        else:
            raise TypeError("Input must be path string or numpy array")
        # Convert to tensor with explicit channel dimension
        tensor = self.transform(Image.fromarray(img))
        tensor = torch.from_numpy(image).permute(2,0,1).float()/255.0 # CxHxW
        if tensor.shape[0]!=3:
            raise ValueError(
                f"Invalid channel dimension: {tensor.shape[0]}"
                "Expected 3-channel RGB image"
            )
        return tensor.to(self.device)

    @staticmethod
    def prepare_for_qat() -> None:
        """Modify model for quantization-aware training"""
        from pytorch_quantization import quant_modules
        quant_modules.initialize()

    def classify_crop(self, image:Union[np.ndarray, Image.Image],
                      bbox: Tuple[float, float ,float, float]) -> Dict[str, Union[int, float]]:
        """
        Classify a cropped region of an image
        Args:
            image: numpy array or PIL Image
            bbox: bounding box coordinates (x1, y1, x2, y2)
        Returns:
            dict with class probabilities
        """
        # Optional: Convert from numpy array to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Crop region using bbox
        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))

        # Preprocess and run inference
        x = self.preprocess_image(crop)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        # Get top prediction
        prob, class_idx = torch.max(probabilities, dim=1)

        return {"class_id":int(class_idx),
                "confidence":float(prob)}

    def get_uncertain_samples(self, pool_loader:DataLoader, top_k:int=100) -> List[Any]:
        """Identify top-k most uncertain samples"""
        uncertainties = []
        with torch.inference_mode():
            for images, _ in pool_loader:
                outputs = self.model(images.to(self.device))
                probs = F.softmax(outputs, dim=1)
                uncertainties.extend((-probs*torch.log2(probs)).sum(dim=1).cpu().numpy()) # Entropy
        indices = np.argsort(uncertainties)[-top_k:]
        return [self.unlabeled_pool[i] for i in indices]

    def active_learning_step(self, pool_loader:DataLoader, human_labeler:Callable) -> None:
        # 1. Get uncertain samples
        samples = self.get_uncertain_samples(pool_loader)
        # 2. Human labeling
        labeled_data = human_labeler(samples)
        # 3. Update training set
        if self.active_learner is not None:
            self.active_learner.update_dataset(labeled_data)
        else:
            raise ValueError("Active learner not initialized")
        # 4. Retrain
        self.train(get_train_transforms(), get_val_transforms())

    def quantize(self) -> None:
        self.model = quantize_dynamic(self.model.to("cpu"),
                                      {nn.Linear, nn.Conv2d},
                                      dtype=torch.qint8)
        self.model.to(self.device)

    def train(self, train_loader:DataLoader, val_loader:Optional[DataLoader],
              epochs:int=100, learning_rate:float=0.001) -> None:
        """
        Train the classifier
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        if self.active_learner:
            train_loader = DataLoader(self.active_learner.current_dataset,
                                      batch_size = 32, shuffle=True, pin_memory=True,
                                      persistent_workers=True)
        class_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        nutrition_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               patience=5, factor=0.5)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer,
                                           class_criterion, nutrition_criterion)

            # Validation
            val_loss, val_accuracy = self._validate_epoch(val_loader, class_criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    def _train_epoch(self, train_loader:DataLoader, optimizer:torch.optim.Optimizer,
                     class_criterion: Callable[[Tensor, Tensor], Tensor],
                     nutrition_criterion: Callable[[Tensor, Tensor], Tensor]) -> float:
        self.model.train()
        total_loss = 0

        for images, (labels, nutrition) in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            nutrition = nutrition.to(self.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = (0.7 * class_criterion(outputs["class"], labels) +
                        0.3 * nutrition_criterion(outputs["nutrition"], nutrition))
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader:DataLoader,
                        criterion:Callable[[Tensor, Tensor], Tensor]) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, (labels, nutrition) in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs["class"], labels)

                total_loss += loss.item()
                _, predicted = outputs["class"].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(val_loader), 100. * correct / total



