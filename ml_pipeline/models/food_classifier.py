from pathlib import Path

import cv2

from PIL import Image
import numpy as np

import torch.nn.functional as F
import torch.cuda
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from torch.quantization import quantize_dynamic
from torchvision.datasets import ImageFolder

from food_detector import FoodDetector
import matplotlib.pyplot as plt

class ActiveLearner:
    def __init__(self, base_dataset, unlabeled_pool):
        self.base_dataset = base_dataset
        self.unlabeled_pool = ImageFolder(unlabeled_pool, transform=base_dataset.transform)
        self.labeled_food = set()
    def update_dataset(self, new_samples):
        self.labeled_food.update(new_samples)
        self.current_dataset = ConcatDataset([self.base_dataset,
                                              Subset(self.unlabeled_pool, list(self.labeled_food))])

    # def get_pool_loader(self, batch_size=64):
    #     return DataLoader(self.unlabeled_pool, batch_size=batch_size)


def display_image(img):
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
def human_labeling_interface(samples):
    labeled = []
    for img, pred in samples:
        display_image(img)
        true_label = input(f"Model Predicted {pred}. Enter correct label: ")
        labeled.append((img, true_label))
    return labeled


class FoodClassifier:
    def __init__(self, model_path:str=None, num_classes:int=256,
                 device:torch.device = None, quantized:bool = None,
                 active_learner = None, label_smoothing=0.1,  **kwargs):
        """
        Initialize food classifier with MobileNetV3
        Args:
            model_path: Path to trained model weights
            num_classes: Number of food classes
            device: Device to run model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        # Initialize model
        self.model = torch.hub.load("pytorch/vision:v0.10.0",
                                    "mobilenet_v3_small", pretrained=True)
        self.model.classifier[-1] = torch.nn.Linear(in_features = self.model.classifier[-1].in_features,
                                                    out_features = num_classes)
        # Load train weights if provided
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path,
                                                  map_location=self.device))
        # Quantization
        if quantized:
            self._fuse_layers()
            self.model = quantize_dynamic(self.model.to("cpu"), # Dynamic quantization works best on CPU
                                          {torch.nn.Linear, torch.nn.Conv2d}, dtype = torch.qint8)
        self.active_learner = active_learner
        self.unlabeled_pool = []
        self.label_smoothing = label_smoothing
        self.model = self.model.to(device)
        self.model.eval()

        # Setup image preprocessing
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std = [0.229, 0.224, 0.225])])

    def preprocess_image(self, image):
        """Preprocess image for classification"""
        # Handle path input
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not read image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Validate numpy array input
        elif isinstance(image, np.ndarray):
             if image.ndim != 3 or image.shape[2] != 3:
                 raise ValueError(f"Invalid image shape: {image.shape}"
                                  "Expected (H, W, 3) RGB array")
             if image.dtype != np.uint8:
                 raise ValueError(f"Invalid dtype: {image.shape}"
                                  "Expected uint8 (0-255 range)")
        else:
            raise TypeError("Input must be path string or numpy array")
        # Convert to tensor with explicit channel dimension
        tensor = torch.from_numpy(image).permute(2,0,1).float()/255.0 # CxHxW
        if tensor.shape[0]!=3:
            raise ValueError(
                f"Invalid channel dimension: {tensor.shape[0]}"
                "Expected 3-channel RGB image"
            )
        return tensor.to(self.device)

    def classify_crop(self, image, bbox):
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

    def _fuse_layers(self):
        """Fuse Conv+BN+ReLU layers for quantization compatibility"""
        for module_name, module in self.model.named_children():
            if "features" in module_name:
                torch.quantization.fuse_modules(module, [["0.0","0.1","0.2"]], # Conv2d + BN + ReLU
                                                inplace = True)
    def calibrate(self, calibration_loader):
        self.model.eval()
        with torch.no_grad():
            for images, _ in calibration_loader:
                _ = self.model(images.to("cpu"))
    def get_uncertain_samples(self, pool_loader, top_k=100):
        """Identify top-k most uncertain samples"""
        uncertainties = []
        with torch.inference_mode():
            for images, _ in pool_loader:
                outputs = self.model(images.to(self.device))
                probs = F.softmax(outputs, dim=1)
                uncertainties.extend((-probs*torch.log2(probs)).sum(dim=1).cpu().numpy()) # Entropy
        indices = np.argsort(uncertainties)[-top_k:]
        return [self.unlabeled_pool[i] for i in indices]

    def active_learning_step(self, pool_loader, human_labeler):
        # 1. Get uncertain samples
        samples = self.get_uncertain_samples(pool_loader)
        # 2. Human labeling
        labeled_data = human_labeler(samples)
        # 3. Update training set
        self.active_learner.update_dataset(labeled_data)
        # 4. Retrain
        self.train(self.active_learner.train_loader, self.active_learner.val_loader)


    def train(self, train_loader, val_loader, epochs:int=100, learning_rate:float=0.001):
        """
        Train the classifier
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        from torch.utils.checkpoint import checkpoint
        def custom_forward(x):
            return self.model(x)
        if self.active_learner:
            train_loader = DataLoader(self.active_learner.current_dataset,
                                      batch_size = 32, shuffle=True)
        class_criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        nutrition_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               patience=5, factor=.5)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            """Training"""
            self.model.train()
            train_loss = 0
            for images, (labels, nutrition) in train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)
                nutrition = nutrition.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = checkpoint(custom_forward, images)
                    # Combine Loss
                    loss = 0.7*class_criterion(outputs["class"], labels)+\
                           0.3*nutrition_criterion(outputs["nutrition"],nutrition)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            # with torch.no_grad():
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = class_criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_accuracy = 100.*correct/total
            val_loss = val_loss/len(val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Epoch {epochs+1}/{epochs}: ")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# For example usage
# if __name__ == "__main__":
#     # Initialize detector and classifier
#     detector = FoodDetector(confidence=0.5)
#     classifier = FoodClassifier(num_classes=256)
#
#     # Load and process an image
#     image_path = "example_food.jpg"
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Detect food items
#     detections = detector.detect(image_rgb)
#
#     # Classify each detection
#     results = []
#     for det in detections:
#         bbox = det['bbox']
#         classification = classifier.classify_crop(image_rgb, bbox)
#
#         results.append({
#             'bbox': bbox,
#             'detection_conf': det['confidence'],
#             'class_id': classification['class_id'],
#             'classification_conf': classification['confidence']
#         })
#
#     print(f"Found {len(results)} food items:")
#     for r in results:
#         print(f"Class {r['class_id']} at {r['bbox']} with confidence {r['classification_conf']:.2f}")


