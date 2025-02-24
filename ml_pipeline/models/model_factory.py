from typing import Tuple

import torch
import torch.nn as nn
import torchvision


def create_backbone(arch:str="mobilenet_v3_small", pretrained:bool=True) -> nn.Sequential:
    """Create feature extraction backbone"""
    if hasattr(torchvision.models, arch):
        model = torchvision.models.__dict__[arch](pretrained=pretrained)
    else:
        raise ValueError(f"Invalid architecture: {arch}")
    # Handle backbone structure
    if arch.startswith("resnet"):
        return nn.Sequential(*list(model.children()))[:-2]
    elif arch.startswith("efficientnet"):
        return model.features
    else: # MobileNetV3/DenseNet etc.
        return nn.Sequential(*list(model.children())[:-1])

def create_multitask_head(in_features: int, num_outputs: int = 4) -> nn.Sequential:
    """
    Create nutrition estimation head.
    Args:
        in_features: Number of input features.
        num_outputs: Number of output nutrition values (default is 4: calories, protein, fat, carbs).
    Returns:
        nn.Sequential: Nutrition estimation head.
    """
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),
        nn.Sigmoid()  # Assumes nutrition values are normalized to [0,1]
    )

def get_feature_dim(backbone: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> int:
    """
    Calculate the number of features output by the backbone.
    Args:
        backbone: The feature extraction backbone.
        input_size: The input size to the backbone (default is (3, 224, 224)).
    Returns:
        int: The number of features in the flattened output.
    """
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        features = backbone(dummy_input)
        if features.dim() != 4:
            raise ValueError("Backbone output must be a 4D tensor (batch, channels, height, width)")
        return features.view(-1).shape[0]