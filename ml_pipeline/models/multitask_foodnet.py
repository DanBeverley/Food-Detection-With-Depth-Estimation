import logging
from typing import Dict, Tuple

import torch
from torch import nn
from model_factory import create_backbone, get_feature_dim, create_multitask_head


class MultiTaskFoodNet(nn.Module):
    """
    A multitask neural network for food image analysis.

    Args:
        num_classes (int): Number of food categories.
        calories_scale (int): Scaling factor for calorie predictions.
        protein_scale (int): Scaling factor for protein predictions.
        arch (str, optional): Backbone architecture (e.g., 'resnet50'). Defaults to 'resnet50'.
        input_size (tuple, optional): Input image size (height, width). Defaults to (224, 224).

    Returns:
        Dict[str, torch.Tensor]: Outputs for classification and nutrition
    """

    def __init__(self, num_classes: int, calories_scale: int,
                 protein_scale: int, arch: str = "resnet50",
                 input_size: Tuple[int, int, int] = (3, 224, 224)) -> None:
        super().__init__()
        self.feature_extractor = create_backbone(arch=arch, pretrained=True)
        self.input_size = input_size

        # Get feature dimensions dynamically
        try:
            in_features = get_feature_dim(self.feature_extractor, input_size=input_size)
        except Exception as e:
            logging.warning(f"Error determining feature dimensions: {e}. Using default value.")
            # Fallback dimensions based on architecture type
            if "resnet" in arch:
                in_features = 2048
            elif "efficientnet" in arch:
                in_features = 1280
            elif "mobilenet" in arch:
                in_features = 1280
            else:
                in_features = 2048

        self.shared_fc = nn.Sequential(
            nn.Flatten(),  # Add explicit flatten for robustness
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(512, num_classes)
        )

        self.portion_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        # Expanded nutrition head with more explicit naming
        self.nutrition_head = create_multitask_head(1024, 4, activation="relu")

        # Initialize scaling factors as registered buffers
        self.register_buffer('calories_scale', torch.tensor([calories_scale], dtype=torch.float))
        self.register_buffer('protein_scale', torch.tensor([protein_scale], dtype=torch.float))

        # Initialize weights with Xavier initialization due to sigmoid
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        # Apply initialization
        self.nutrition_head.apply(init_weights)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Dictionary with keys:
            - 'class': Food class logits [batch_size, num_classes]
            - 'nutrition': Nutrition values [batch_size, 4]
              - nutrition[:,0]: Calories (kcal)
              - nutrition[:,1]: Protein (g)
              - nutrition[:,2]: Fat (g)
              - nutrition[:,3]: Carbohydrates (g)
            - 'portion': Portion size estimate [batch_size]
        """
        try:
            # Handle input size mismatches
            _, c, h, w = x.shape
            if (c, h, w) != self.input_size:
                raise ValueError(f"Input size mismatch: got {(c, h, w)}, expected {self.input_size}")

            # Extract features
            features = self.feature_extractor(x)

            # Handle different output shapes from backbone
            if features.dim() != 4 and features.dim() != 2:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

            # Apply shared layers
            shared = self.shared_fc(features)

            # Get task-specific outputs
            class_output = self.class_head(shared)

            # Process nutrition outputs with proper scaling
            nutrition_raw = self.nutrition_head(shared)

            # Scale nutrition values to realistic ranges
            # [calories, protein, fat%, carbs%]
            nutrition = torch.stack([
                nutrition_raw[:, 0] * self.calories_scale,  # Calories in kcal
                nutrition_raw[:, 1] * self.protein_scale,  # Protein in grams
                nutrition_raw[:, 2] * 100,  # Fat percentage 0-100%
                nutrition_raw[:, 3] * 100  # Carbs percentage 0-100%
            ], dim=1)

            return {
                "class": class_output,
                "nutrition": nutrition,
                "portion": self.portion_head(shared).squeeze()
            }

        except RuntimeError as e:
            logging.error(f"Forward pass failed: {e}")
            raise

