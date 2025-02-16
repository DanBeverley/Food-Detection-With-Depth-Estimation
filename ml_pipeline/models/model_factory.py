import torch
import torch.nn as nn

def create_backbone(arch="mobilenet_v3_small", pretrained=True):
    """Create feature extraction backbone"""
    model = torch.hub.load("pytorch/vision:v0.10.0", arch, pretrained=pretrained)
    return nn.Sequential(*list(model.children())[:-1])

def create_classification_head(in_features, num_classes):
    """Create classification head with adaptive features"""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, num_classes)
    )

def create_multitask_head(in_features):
    """Create nutrition estimation head"""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 4),
        nn.Sigmoid()
    )