import torch
from torch import nn
class MultiTaskFoodNet(nn.Module):
    def __init__(self, num_classes:int = 256):
        super().__init__()
        self.backbone = torch.hub.load("pytorch/vision", "mobilenet_v3_small",
                                       pretrained = True)
        in_features = self.backbone.classifier[-1].in_features

        # Classification head
        self.class_head = nn.Sequential(nn.Linear(in_features, 512),
                                        nn.ReLU(), nn.Linear(512, num_classes))
        # Nutrition estimation head
        self.nutrition_head = nn.Sequential(nn.Linear(in_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 4)) #[calories, protein, fat, carbs]
    def forward(self, x):
        features = self.backbone(x)
        return {"class":self.class_head(features),
                "nutrition":self.nutrition_head(features)}
