import torch
from torch import nn
from model_factory import create_backbone, create_multitask_head
class MultiTaskFoodNet(nn.Module):
    def __init__(self, num_classes:int = 256):
        super().__init__()
        self.feature_extractor = create_backbone()
        self.nutrition_head = create_multitask_head(in_features=576)
        self.calories_scale = nn.Parameter(torch.tensor([500.0]))
        self.protein_scale = nn.Parameter(torch.tensor([100.0]))
        self.backbone = torch.hub.load("pytorch/vision", "mobilenet_v3_small",
                                       pretrained = True)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        # Get actual feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.feature_extractor(dummy_input)
            in_features = features.view(-1).shape[0]
        self.shared_fc = nn.Sequential(nn.Linear(in_features, 1024),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        # Classification head
        self.class_head = nn.Sequential(nn.Linear(1024, 512),
                                        nn.ReLU(), nn.Linear(512, num_classes))
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten
        nutrition = self.nutrition_head(features)
        # Scale outputs to realistic ranges
        nutrition[:,0] *= self.calories_scale
        nutrition[:,1] *= self.protein_scale

        return {"class":self.class_head(features),
                "nutrition":nutrition}
