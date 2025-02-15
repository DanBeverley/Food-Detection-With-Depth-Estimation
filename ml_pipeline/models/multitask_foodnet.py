import torch
from torch import nn
class MultiTaskFoodNet(nn.Module):
    def __init__(self, num_classes:int = 256):
        super().__init__()
        self.backbone = torch.hub.load("pytorch/vision", "mobilenet_v3_small",
                                       pretrained = True)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        # Get actual feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.feature_extractor(dummy_input)
            in_features = features.view(-1).shape[0]

        # Classification head
        self.class_head = nn.Sequential(nn.Linear(in_features, 512),
                                        nn.ReLU(), nn.Linear(512, num_classes))
        # Nutrition estimation head
        self.nutrition_head = nn.Sequential(nn.Linear(in_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 4),
                                            nn.Sigmoid()) #[calories, protein, fat, carbs]
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten
        return {"class":self.class_head(features),
                "nutrition":self.nutrition_head(features)}
