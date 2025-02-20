import json
import logging
import csv
from pathlib import Path
from typing import Callable

import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision

from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
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
        self.image_dir = self.root_dir/image_dir
        self.mask_dir = self.root_dir/mask_dir
        self.transform = transform or get_gan_transforms()

        self.mask_paths = {p.stem:p for p in self.mask_dir.glob(".png")}
        self.image_paths = [p for p in self.image_dir.rglob(f"*{image_ext}")
                            if p.stem in self.mask_paths]
        if not self.image_paths:
            raise ValueError(f"No matching images found in {image_dir}")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        mask = cv2.imread(str(self.mask_paths[self.image_paths[idx].stem]))
        return self.transform(image=np.array(img), mask = np.array(mask))

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
    @staticmethod
    def _init_weight(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc:int=3, ndf:int=64):
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
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.main(x).view(-1)

# Gradient Penalty Function
# ------------------

def compute_gradient_penalty(D:nn.Module, real_samples:torch.Tensor,
                             fake_samples:torch.Tensor,
                             device:torch.cuda.device,
                             lambda_gp:float=10.0)->torch.Tensor:
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

class GANTrainer:
    def __init__(self, device:torch.cuda.device, nz:int=100, lr:float=0.0002,
                 beta1:float=0.5, nutrition_mapper = None):
        self.device = device
        self.nz = nz
        self.nutrition_mapper = nutrition_mapper
        self.netG = Generator(nz = nz).to(device=device)
        self.netD = Discriminator().to(device=device)

        self.optimG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimG, "min", patience = 5)
        self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimD, "min", patience = 5)

        self._setup_logging()
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
    def _setup_logging(self):
        """Initialize logging and checkpointing"""
        self.sample_dir = Path("gan_samples")
        self.sample_dir.mkdir(exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir="output/")

        # Metadata file for estimation compatibility (CSV with header)
        self.metadata_file = self.sample_dir/"metadata.csv"
        if not self.metadata_file.exists():
            self._initialize_metadata()

    def _initialize_metadata(self):
        """Initialize the metadata CSV file with headers"""
        headers = [
            "epoch",
            "sample_filename",
            "synthetic",
            "calories_estimate",
            "food_type",
            "portion_size",
            "generation_parameters",
            "model_version"
        ]
        try:
            with open(self.metadata_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)

            logging.info(f"Initialized metadata file at {self.metadata_file}")
        except Exception as e:
            logging.error(f"Failed to initialize metadata file: {e}")
            raise RuntimeError("Metadata initialization failed")

    def _save_metadata(self, epoch:int, sample_filename:str,
                       generation_params:dict=None):
        """
        Save metadata for generated samples with enhanced tracking

        Args:
            epoch: Current training epoch
            sample_filename: Name of the generated sample file
            generation_params: Dictionary of generation parameters (optional)
        """
        food_categories = ["rice", "sushi", "ramen", "tempura"]
        fake_label = np.random.choice(food_categories)
        if self.nutrition_mapper:
            nutrition = self.nutrition_mapper.map_food_label_to_nutrition(fake_label)
        else:
            nutrition = NutritionMapper.get_default_nutrition()

        if generation_params is None:
            generation_params = {}
        # Generate realistic portion estimates based on food type
        fake_label = "synthetic_food"
        nutrition = self.nutrition_mapper.map_food_label_to_nutrition(fake_label)

        bbox_area = np.random.randint(500,2000)
        density = self.nutrition_mapper.get_density(fake_label) if self.nutrition_mapper else 0.8
        portion = bbox_area * density
        calories = portion * (nutrition.get("calories_per_ml", 0.5) if nutrition else 0.5)

        row_data = {
            "epoch": epoch,
            "sample_filename": sample_filename,
            "synthetic": True,
            "calories_estimate": round(calories, 2),
            "food_type": fake_label,
            "portion_size": round(portion, 2),
            "generation_parameters": json.dumps(generation_params),
            "model_version": "1.0"
        }
        try:
            with open(self.metadata_file, "a", newline = "") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(row_data.keys()))
                writer.writerow(row_data)
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            pass # Continue training if saving metadata failed

    def train(self, dataloader:DataLoader, epochs:int):
        best_loss = float("inf")
        for epoch in range(epochs):
            epoch_losses = self._train_epoch(dataloader, epoch, epochs)
            # Learning rate adjustment
            avg_loss = (epoch_losses["G"] + epoch_losses["D"])/2
            self.schedulerG.step(avg_loss)
            self.schedulerD.step(avg_loss)
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(epoch, is_best=True)
            self._log_epoch(epoch, epoch_losses)

    def _train_step(self, real_imgs:torch.Tensor)->dict:
        """
        Execute a single training step for both Generator and Discriminator
        Args:
            real_imgs: Batch of real images
        Returns:
            dict: Losses for Generator and Discriminator
        """
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)
        # Train Discriminator
        self.netD.zero_grad()
        with autocast():
            # Real images
            output_real = self.netD(real_imgs)
            loss_real = self.criterion(output_real, real_labels)
            # Fake images
            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake_imgs = self.netG(noise)
            output_fake = self.netD(fake_imgs.detach())
            loss_fake = self.criterion(output_fake, fake_labels)
            # Gradient penalty
            gp = compute_gradient_penalty(self.netD, real_imgs, fake_imgs, self.device, lambda_gp=10.0)
            loss_D = (loss_real + loss_fake)*0.5 + gp * 10.0
        # Mixed precision backward pass for discriminator
        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.optimD)
        # Train Generator
        self.netG.zero_grad()
        with autocast():
            output = self.netD(fake_imgs)
            loss_G = self.criterion(output, real_labels)

        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optimG)
        self.scaler.update()
        return {
            'G': loss_G.item(),
            'D': loss_D.item(),
            'GP': gp.item()
        }

    def _train_epoch(self, dataloader:DataLoader, epoch:int, total_epochs:int):
        self.netG.train()
        self.netD.train()
        epoch_losses = {"G":0., "D":0.}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for batch_idx, (index, real_imgs) in enumerate(progress_bar):
            losses = self._train_step(real_imgs.to(self.device))
            for k, v in losses.items():
                epoch_losses[k] += v
            if batch_idx % 100 == 0:
                self._log_step(batch_idx, losses)
        return {k: v/len(dataloader) for k, v in epoch_losses.items()}

    def _log_step(self, batch_idx: int, losses: dict):
        """
        Log training metrics for a single step

        Args:
            batch_idx: Current batch index
            losses: Dictionary of loss values
        """
        # Log to tensorboard
        for name, value in losses.items():
            self.writer.add_scalar(f'step/loss_{name}', value, batch_idx)
        # Log generator/discriminator ratio
        g_d_ratio = losses['G'] / losses['D'] if losses['D'] != 0 else 0
        self.writer.add_scalar('step/G_D_ratio', g_d_ratio, batch_idx)
        # Log gradient penalty
        if 'GP' in losses:
            self.writer.add_scalar('step/gradient_penalty', losses['GP'], batch_idx)

    def _log_epoch(self, epoch: int, epoch_losses: dict):
        """
        Log training metrics for an entire epoch
        Args:
            epoch: Current epoch number
            epoch_losses: Dictionary of average loss values for the epoch
        """
        # Log average losses
        for name, value in epoch_losses.items():
            self.writer.add_scalar(f'epoch/loss_{name}', value, epoch)
        # Generate and log sample images
        with torch.no_grad():
            fake_images = self.netG(self.fixed_noise)
            grid = torchvision.utils.make_grid(fake_images, normalize=True)
            self.writer.add_image('generated_samples', grid, epoch)
        # Log learning rates
        self.writer.add_scalar('epoch/lr_G', self.optimG.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('epoch/lr_D', self.optimD.param_groups[0]['lr'], epoch)
        # Save samples
        sample_filename = f"epoch_{epoch + 1}.png"
        self._save_samples(fake_images, sample_filename)
        self._save_metadata(epoch, sample_filename, {
            'losses': epoch_losses,
            'lr_G': self.optimG.param_groups[0]['lr'],
            'lr_D': self.optimD.param_groups[0]['lr']
        })

    def _save_samples(self, fake_images: torch.Tensor, filename: str):
        """
        Save generated sample images

        Args:
            fake_images: Tensor of generated images
            filename: Name of the output file
        """
        output_path = self.sample_dir / filename
        try:
            torchvision.utils.save_image(
                fake_images,
                output_path,
                normalize=True,
                nrow=8
            )
        except Exception as e:
            logging.error(f"Failed to save samples: {e}")

    def _save_checkpoint(self, epoch:int, is_best:bool=False):
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.netG.state_dict(),
            'D_state_dict': self.netD.state_dict(),
            'optimG_state_dict': self.optimG.state_dict(),
            'optimD_state_dict': self.optimD.state_dict(),
        }
        path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        if is_best:
            best_path = "best_model.pt"
            torch.save(checkpoint, best_path)