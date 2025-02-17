import torch
from torch import nn
from model_factory import create_backbone, get_feature_dim, create_multitask_head

class MultiTaskFoodNet(nn.Module):
    def __init__(self, num_classes:int = 256):
        super().__init__()
        self.feature_extractor = create_backbone()
        in_features = get_feature_dim(self.feature_extractor)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.nutrition_head = create_multitask_head(1024)

        # Initialize scaling factors as registered buffers instead of parameters
        self.register_buffer('calories_scale', torch.tensor([500.0]))
        self.register_buffer('protein_scale', torch.tensor([100.0]))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten
        shared = self.shared_fc(features)
        # Scale outputs to realistic ranges
        nutrition = self.nutrition_head(shared)
        nutrition[:,0] *= self.calories_scale
        nutrition[:,1] *= self.protein_scale

        return {"class":self.class_head(shared),
                "nutrition":nutrition}
