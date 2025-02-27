from pathlib import Path
from typing import Any, Union, Iterable, Tuple, List, Optional, Dict, Callable

import cv2

from PIL import Image
import numpy as np

from torch import nn, Tensor
import torch.nn.functional as F
import torch.cuda
from ml_pipeline.data_processing.dataset_loader import UECFoodDataset
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from torch.quantization import quantize_dynamic
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose
from torch.utils.tensorboard import SummaryWriter
from multitask_foodnet import MultiTaskFoodNet
import matplotlib.pyplot as plt

pool_dataset = ImageFolder("unlabeled_pool_path", transform=Compose([ToTensor()]))
pool_loader = DataLoader(pool_dataset, batch_size=32, shuffle = False)

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
                 calories_scale:int = 500, protein_scale:int = 100,
                 class_loss_weight:float=0.7, nutrition_loss_weight:float=0.3) -> None:
        """
        Initialize food classifier with MobileNetV3
        Args:
            model_path: Path to trained model weights
            num_classes: Number of food classes
            device: Device to run model on
        """
        self.class_loss_weight = class_loss_weight
        self.nutrition_loss_weight = nutrition_loss_weight
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
        else:
            self.model = self.model.to(self.device)
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

    def preprocess_image(self, image:Union[str, Path, np.ndarray, Image.Image],
                         color_format:str="RGB") -> Tensor:
        """Preprocess image for classification
           Args:
                image: Input image (path, numpy array, or PIL Image).
                color_format: Color format of numpy array input ("RGB" or "BGR"). Default is "RGB".
           Returns:
                Preprocessed tensor."""
        # Handle different input types
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.dtype != np.uint8:
                raise ValueError("Expected image of dtype uint8 in the 0-255 range")
            if img.ndim == 2:
                import warnings
                warnings.warn("Grayscale image detected. Converting to RGB, but model may expect color images.")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                if color_format == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        # Convert to tensor with explicit channel dimension
        tensor = self.transform(Image.fromarray(img))
        if tensor.shape[0]!=3:
            # First check, then handle, then raise if needed
            if tensor.shape[0] > 3:
               tensor = tensor[:3] # Force 3 channels
            else:
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
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)},"
                            f"expected numpy array, PIL image, or path")
        # Crop region using bbox
        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))

        # Preprocess and run inference
        x = self.preprocess_image(crop)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            class_output = output["class"]
            probabilities = torch.nn.functional.softmax(class_output, dim=1)
        # Get top prediction
        prob, class_idx = torch.max(probabilities, dim=1)
        return {"class_id":int(class_idx.item()),
                "confidence":float(prob.item()),
                "nutrition":output["nutrition"][0].cpu().tolist()}

    def get_uncertain_samples(self, pool_loader:DataLoader, num_samples:int) -> List[Any]:
        """Identify top-k most uncertain samples"""
        uncertainties = []
        sample_paths = []
        with torch.inference_mode():
            for batch_idx, (images, _) in enumerate(pool_loader):
                if not isinstance(images, torch.Tensor):
                    raise TypeError(f"Expected images as tensors, got {type(images)}")
                outputs = self.model(images.to(self.device))["class"]
                probs = F.softmax(outputs, dim=1)
                batch_uncertainties = -torch.sum(probs * torch.log(probs + 1e-10),dim=1)
                uncertainties.extend(batch_uncertainties.cpu().tolist())
                sample_paths.extend([f"sample_{i}" for i in range(len(images))])
        # Sort by uncertainty and return top num_samples
        indices = np.argsort(uncertainties)[-num_samples:]
        return [sample_paths[i] for i in indices]

    def active_learning_step(self, pool_loader:DataLoader, human_labeler:Callable,
                             val_split:float=0.2) -> None:
        """
        Perform one step of active learning

        Args:
            pool_loader: DataLoader for the unlabeled pool
            human_labeler: Function to get human labels for samples
            val_split: Percentage of labeled data to use for validation
        """
        # 1. Get uncertain samples
        samples = self.get_uncertain_samples(pool_loader = pool_loader, num_samples = 100)
        # 2. Human labeling
        labeled_data = human_labeler(samples)
        # 3. Update training set
        if self.active_learner is not None:
            self.active_learner.update_dataset(labeled_data)
        else:
            raise ValueError("Active learner not initialized")
        # 4. Splitting
        dataset = self.active_learner.current_dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        import random
        random.shuffle(indices)

        val_size = int(np.floor(val_split*dataset_size))
        train_indices, val_indices = indices[val_size:], indices[:val_size]
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=32, shuffle=False)
        self.train(train_loader, val_loader)

    def quantize(self) -> None:
        self.model = quantize_dynamic(self.model.to("cpu"),
                                      {nn.Linear, nn.Conv2d},
                                      dtype=torch.qint8)
        self.model.to(self.device)

    def train(self, train_loader:DataLoader, val_loader:Optional[DataLoader],
              epochs:int=100, learning_rate:float=0.001, patience:int=5) -> None:
        """
        Train the classifier
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.writer = SummaryWriter()
        if self.active_learner and train_loader is None:
            train_loader = DataLoader(self.active_learner.current_dataset,
                                      batch_size = 32, shuffle=True, pin_memory=True,
                                      persistent_workers=True)
        class_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        nutrition_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               patience=5, factor=0.5)
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer,
                                           class_criterion, nutrition_criterion)

            # Validation
            val_class_loss, val_nutrition_loss, val_accuracy, nutrition_error = self._validate_epoch(val_loader, class_criterion, nutrition_criterion)
            val_loss = self.class_loss_weight*val_class_loss+self.nutrition_loss_weight*val_nutrition_loss
            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                epochs_no_improve+=1
                if epochs_no_improve>=patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Class Loss: {val_class_loss:.4f} |"
                  f"Val Nutrition Loss: {val_nutrition_loss:.4f} |"
                  f"Val Accuracy: {val_accuracy:.2f}%")

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
                loss = (self.class_loss_weight * class_criterion(outputs["class"], labels) +
                        self.nutrition_loss_weight * nutrition_criterion(outputs["nutrition"], nutrition))
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader:DataLoader, class_criterion:Callable[[Tensor, Tensor], Tensor],
                        nutrition_criterion:Callable[[Tensor, Tensor], Tensor]) -> tuple[float, float, float, Any]:
        self.model.eval()
        total_class_loss = 0
        total_nutrition_loss = 0
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, (labels, nutrition) in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                nutrition = nutrition.to(self.device)
                outputs = self.model(images)
                class_loss = class_criterion(outputs["class"], labels)
                nutrition_loss = nutrition_criterion(outputs["nutrition"], nutrition)

                total_class_loss += class_loss.item()
                total_nutrition_loss += nutrition_loss.item()
                _, predicted = outputs["class"].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_class_loss = total_class_loss/len(val_loader)
        avg_nutrition_loss = total_nutrition_loss/len(val_loader)
        accuracy = 100.*correct/total
        nutrition_error = (outputs["nutrition"]-nutrition).abs().mean().item()
        return avg_class_loss, avg_nutrition_loss, accuracy, nutrition_error



