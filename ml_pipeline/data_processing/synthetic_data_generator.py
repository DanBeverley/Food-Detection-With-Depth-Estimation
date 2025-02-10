import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from dataset_loader import UECFoodDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# GAN CONFIGURATION
IMG_SIZE = 224
LATENT_DIM = 256
BATCH_SIZE = 32
EPOCHS = 100
SAVE_INTERVAL = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAN SPECIFIC TRANSFORMATION
gan_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                           A.Normalize(mean=[.5, .5, .5],
                                       std =[.5, .5 ,.5]),
                                       ToTensorV2()])
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)