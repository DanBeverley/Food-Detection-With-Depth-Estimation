import os
import csv
from pathlib import Path
from typing import Callable

import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision
from ml_pipeline.utils.transforms import get_gan_transforms
from torch import optim
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import spectral_norm
from tqdm import tqdm

class FoodGANDataset(Dataset):
    def __init__(self, root_dir:str,image_dir:str,
                 mask_dir:str, transform:Callable=None, image_ext:str=".jpg"):
        self.root_dir = Path(root_dir)
        self.mask_paths = {p.stem:p for p in (self.root_dir / mask_dir).glob("*.png")}
        self.image_paths = [p for p in (self.root_dir / image_dir).glob(f"*{image_ext}") if p.stem in self.mask_paths]
        self.transform = transform or get_gan_transforms()
        # Efficient recursive scan
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(image_ext.lower()):
                    self.image_paths.append(os.path.join(root, file))
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx:int):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
    def load_item(self, idx):
        return self.__getitem__(idx)

# GAN Architecture
# ----------------------

class Generator(nn.Module):
    def __init__(self, nz:int=100, ngf:int=64, nc:int=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(self._init_weight)
    def _init_weight(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        # Use spectral normalization for each conv layer for stability.
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()  # Even though WGAN-GP usually doesn't use Sigmoid, here we add gradient penalty on top of BCE.
        )
    def forward(self, x):
        return self.main(x).view(-1)

# Gradient Penalty Function
# ------------------

def compute_gradient_penalty(D:nn.Module, real_samples, fake_samples,
                             device:torch.cuda.device, lambda_gp:float=10.0):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device = device)
    interpolates = (alpha*real_samples + ((1-alpha)*fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    ones = torch.ones(d_interpolates.size(), device = device)
    gradients = autograd.grad(outputs = d_interpolates, inputs = interpolates,
                              grad_outputs = ones, create_graph = True,
                              retain_graph = True, only_inputs = True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
    return gradient_penalty

# Training Loop with Mixed Precision, Gradient Penalty and TensorBoard Logging
# -------------------

class GANTrainer:
    def __init__(self, device:torch.device, nz:int=100, lr:float=0.0002,
                 beta1:float=0.5, nutrition_mapper = None):
        self.device = device
        self.nz = nz
        self.nutrition_mapper = nutrition_mapper
        self.netG = Generator(nz = nz).to(device=device)
        self.netD = Discriminator().to(device=device)

        self.optimG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        self.sample_dir = "gan_samples"
        os.makedirs(self.sample_dir, exist_ok = True)

        self.writer = SummaryWriter(log_dir="output/")

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Metadata file for estimation compatibility (CSV with header)
        self.metadata_file = os.path.join(self.sample_dir, "metadata.csv")
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch", "sample_filename", "synthetic", "calories_estimate"])
    def _save_metadata(self, epoch:int, sample_filename:str):
        """Append metadata info for generated sample
           Add a placeholder for calories estimate (-1 indicates 'unknown')"""
        # Generate realistic portion estimates based on food type
        fake_label = "synthetic_food"
        nutrition = self.nutrition_mapper.map_food_label_to_nutrition(fake_label)
        # Get default values if no nutrition data
        density = self.nutrition_mapper.get_density(fake_label)
        calories_per_ml = nutrition.get("calories_per_ml",
                                        0.5) if nutrition else 0.5 # 0.5 cal/ml fallback

        bbox_area = np.random.randint(500,2000)
        portion = bbox_area*density

        with open(self.metadata_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, sample_filename, True,
                             portion*calories_per_ml])

    def train(self, dataloader, epochs):
        lambda_gp = 10.0

        global_step = 0
        for epoch in range(epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for real_imgs in progress_bar:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                # Train Discriminator
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)

                self.netD.zero_grad()
                with autocast():
                    # Real images
                    output_real = self.netD(real_imgs)
                    loss_real = self.criterion(output_real, real_labels)

                    # Fake images
                    noise = torch.randn(batch_size, self.fixed_noise.shape[1], 1, 1, device = self.device)
                    fakes = self.netG(noise)
                    output_fake = self.netD(fakes.detach())
                    loss_fake = self.criterion(output_fake, fake_labels)

                    gp = compute_gradient_penalty(self.netD, real_imgs,
                                                  fakes, self.device, lambda_gp)
                    loss_D = (loss_real+loss_fake)*.5 + lambda_gp*gp
                # Mixed Precision backward pass for discriminator
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimD)

                # Train Generator
                self.netG.zero_grad()
                output = self.netD(fakes)
                lossG = self.criterion(output, real_labels)
                lossG.backward()
                self.optimG.step()

                progress_bar.set_postfix(loss_G=lossG.item(), loss_D=loss_D.item())

            self._save_samples(epoch)
            self._save_checkpoint(epoch)

    def _save_samples(self, epoch):
        with torch.no_grad():
            fake = self.netG(self.fixed_noise)
        torchvision.utils.save_image(fake, f"{self.sample_dir}/epoch_{epoch+1}.png",
                                     nrow=8, normalize=True)

    def _save_checkpoint(self, epoch):
        torch.save(self.netG.state_dict(), f"G_epoch_{epoch + 1}.pth")
        torch.save(self.netD.state_dict(), f"D_epoch_{epoch + 1}.pth")
# Execution
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset with optimized loading
    dataset = FoodGANDataset("path/to/UECFOOD256")
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Training
    trainer = GANTrainer(device)
    trainer.train(dataloader, epochs=50)