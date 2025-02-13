from pathlib import Path
from PIL import Image
import numpy as np

import torch.cuda
from torchvision import transforms


class FoodClassifier:
    def __init__(self, model_path:str=None, num_classes:int=256,
                 device:torch.device = None):
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
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

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

    def train(self, train_loader, val_loader, epochs:int=100, learning_rate:float=0.001):
        """
        Train the classifier
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               patience=5, factor=.5)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            """Training"""
            self.model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

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

