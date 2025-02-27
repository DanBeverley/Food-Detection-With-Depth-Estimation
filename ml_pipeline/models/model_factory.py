import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision


def create_backbone(arch:str="mobilenet_v3_small", pretrained:bool=True) -> nn.Sequential:
    """
    Create feature extraction backbone with support for newer model versions

    Args:
        arch: Architecture name (mobilenet_v3_small, resnet50, etc)
        pretrained: Whether to use pretrained weights

    Returns:
        Feature extraction backbone as nn.Sequential
    """
    # For backward compatibility with older torchvision versions
    weights_param = "pretrained" if hasattr(torchvision.models, "__version__") and \
                                    int(torchvision.models.__version__.split('.')[0]) < 0.13 else "weights"

    # Handle deprecated 'pretrained' parameter in newer torchvision
    if weights_param == "weights" and pretrained:
        import inspect
        model_fn = getattr(torchvision.models, arch, None)
        if model_fn is not None:
            # Get the appropriate weights enum
            signature = inspect.signature(model_fn)
            if "weights" in signature.parameters:
                weights_enum_name = f"{arch.upper().replace('_', '')}_Weights.IMAGENET1K_V1"
                # Try to resolve the weights enum
                try:
                    weights_module = getattr(torchvision.models, f"{arch}_weights", None)
                    if weights_module is not None:
                        weights_enum = getattr(weights_module, "IMAGENET1K_V1", None)
                        kwargs = {"weights": weights_enum}
                    else:
                        kwargs = {"weights": "IMAGENET1K_V1"}
                except (AttributeError, ImportError):
                    kwargs = {weights_param: pretrained}
            else:
                kwargs = {weights_param: pretrained}
        else:
            kwargs = {weights_param: pretrained}
    else:
        kwargs = {weights_param: pretrained}

    # Create the model
    try:
        if hasattr(torchvision.models, arch):
            model = getattr(torchvision.models, arch)(**kwargs)
        else:
            raise ValueError(f"Invalid architecture: {arch}")

        # Handle different backbone structures based on architecture type
        if arch.startswith("resnet"):
            return nn.Sequential(*list(model.children())[:-2])  # Remove avg pool and fc
        elif arch.startswith("efficientnet"):
            if hasattr(model, "features"):
                return model.features
            else:
                # For newer EfficientNet implementations
                layers = list(model.children())
                # Remove classifier head
                return nn.Sequential(*layers[:-1])
        elif "mobilenet" in arch:
            if hasattr(model, "features"):
                return model.features
            else:
                # For newer MobileNet implementations
                layers = list(model.children())
                return nn.Sequential(*layers[:-1])  # Remove classifier
        elif "densenet" in arch:
            if hasattr(model, "features"):
                return model.features
            else:
                # For newer DenseNet implementations
                features = nn.Sequential(
                    model.conv0, model.norm0, model.relu0, model.pool0,
                    model.denseblock1, model.transition1,
                    model.denseblock2, model.transition2,
                    model.denseblock3, model.transition3,
                    model.denseblock4, model.norm5
                )
                return features
        else:
            # Generic fallback
            layers = list(model.children())
            return nn.Sequential(*layers[:-1])  # Remove final classification layer
    except Exception as e:
        raise ValueError(f"Error creating backbone {arch}: {e}")

def create_multitask_head(in_features: int, num_outputs: int = 4,
                          activation:str="sigmoid") -> nn.Sequential:
    """
    Create nutrition estimation head.
    Args:
        in_features: Number of input features.
        num_outputs: Number of output nutrition values (default is 4: calories, protein, fat, carbs).
    Returns:
        nn.Sequential: Nutrition estimation head.
    """
    activation_layer = nn.Sigmoid() if activation=="sigmoid" else nn.ReLU() if activation == "relu" else None
    layers = [nn.Linear(in_features, 256),
              nn.ReLU(), nn.Linear(256, num_outputs)]
    if activation_layer:
        layers.append(activation_layer)
    return nn.Sequential(*layers)

def get_feature_dim(backbone: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> int:
    """
    Calculate the number of features output by the backbone.
    Args:
        backbone: The feature extraction backbone.
        input_size: The input size to the backbone (default is (3, 224, 224)).
    Returns:
        int: The number of features in the flattened output.
    """
    try:
        with torch.no_grad():
            # Add batch dimension
            dummy_input = torch.randn(1, *input_size)
            features = backbone(dummy_input)

            # Handle different output types
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Use last output for multi-output backbones

            # Check output dimension
            if features.dim() == 4:
                # For conv features: [batch, channels, height, width]
                batch_size, channels, height, width = features.shape
                return channels * height * width
            elif features.dim() == 2:
                # For already flattened features: [batch, features]
                return features.shape[1]
            elif features.dim() == 3:
                # For sequence outputs: [batch, seq_len, features]
                return features.shape[1] * features.shape[2]
            else:
                raise ValueError(f"Unexpected feature dimension: {features.dim()}")
    except Exception as e:
        logging.error(f"Error determining feature dimensions: {e}")
        # Fallback to a common dimension
        return 2048