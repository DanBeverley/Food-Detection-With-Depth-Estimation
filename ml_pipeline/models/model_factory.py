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

def create_multitask_head(in_features:int) -> nn.Sequential:
    """Create nutrition estimation head"""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 4),
        nn.Sigmoid()
    )

def get_feature_dim(backbone:nn.Module, input_size:Tuple[int, int, int]=(3, 224, 224)):
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        features = backbone(dummy_input)
        return features.view(-1).shape[0]