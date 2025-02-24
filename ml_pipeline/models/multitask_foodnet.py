import logging
from typing import Dict

import torch
from torch import nn
from model_factory import create_backbone, get_feature_dim, create_multitask_head

class MultiTaskFoodNet(nn.Module):
    """A multitask neural network for food image analysis.
    Args:
        num_classes (int): Number of food categories.
        calories_scale (int): Scaling factor for calorie predictions.
        protein_scale (int): Scaling factor for protein predictions.
        arch (str, optional): Backbone architecture (e.g., 'resnet50'). Defaults to 'resnet50'.
        input_size (tuple, optional): Input image size (height, width). Defaults to (224, 224).

    Returns:
        Dict[str, torch.Tensor]: Outputs for classification, nutrition (calories, protein, fat, carbs), and portion size.
    """
    def __init__(self, num_classes:int, calories_scale:int,
                 protein_scale:int, arch: str = "resnet50", input_size: tuple = (224, 224)) -> None:
        super().__init__()
        self.feature_extractor = create_backbone()
        in_features = get_feature_dim(self.feature_extractor)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.class_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.portion_head = nn.Sequential(nn.Linear(1024, 256),
                                          nn.ReLU(),
                                          nn.Linear(256, 1),
                                          nn.Sigmoid())
        self.nutrition_head = create_multitask_head(1024)

        # Initialize scaling factors as registered buffers instead of parameters
        self.register_buffer('calories_scale', torch.tensor([calories_scale]))
        self.register_buffer('protein_scale', torch.tensor([protein_scale]))
        def init_weights(m:nn.Module) -> None:
            if isinstance(m ,nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.fill_(0.01)
        self.apply(init_weights)
    def forward(self, x:torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1) # Flatten
            shared = self.shared_fc(features)
            # Scale outputs to realistic ranges
            nutrition = self.nutrition_head(shared)
            nutrition = torch.clamp(nutrition, min=0)
            nutrition = torch.stack([
                nutrition[:, 0] * self.calories_scale,
                nutrition[:, 1] * self.protein_scale,
                torch.clamp(nutrition[:, 2], 0, 100), # Fat percentage,
                torch.clamp(nutrition[:, 3],0 , 100) # Carbs Percentage
            ], dim = 1)
            return {"class":self.class_head(shared),
                    "nutrition":nutrition,
                    "portion": self.portion_head(shared).squeeze()}
        except RuntimeError as e:
            logging.error(f"Forward pass failed: {e}")
            raise

