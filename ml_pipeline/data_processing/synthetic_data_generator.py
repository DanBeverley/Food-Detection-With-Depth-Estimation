import os
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import UECFoodDataset

from tqdm import tqdm
from PIL import Image

class FoodGANDataset(Dataset):
    def __init__(self, root_dir:str, transform:transforms=None, image_ext:str=".jpg"):
        self.root_dir = root_dir
        self.transform = transform or transform.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5), (.5,.5,.5))
        ])
        self.image_paths = []

        # Efficient recursive scan
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(image_ext.lower()):
                    self.image_paths.append(os.path.join(root, file))
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx:int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

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
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)


# Training Loop
# -------------------

class GANTrainer:
    def __init__(self, device:torch.device, nz:int=100, lr:float=0.0002,
                 beta1:float=0.5):
        self.device = device
        self.netG = Generator().to(device=device)
        self.netD = Discriminator().to(device=device)

        self.optimG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        self.sample_dir = "gan_samples"
        os.makedirs(self.sample_dir, exist_ok = True)

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for real_imgs in progress_bar:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                # Train Discriminator
                self.netD.zero_grad()
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)

                # Real images
                output_real = self.netD(real_imgs)
                loss_real = self.criterion(output_real, real_labels)

                # Fake images
                noise = torch.randn(batch_size, 100, 1, 1, device = self.device)
                fakes = self.netG(noise)
                output_fake = self.netD(fakes.detach())
                loss_fake = self.criterion(output_fake, fake_labels)

                lossD = (loss_real+loss_fake)*.5
                lossD.backward()
                self.optimD.step()

                # Train Generator
                self.netG.zero_grad()
                output = self.netD(fakes)
                lossG = self.criterion(output, real_labels)
                lossG.backward()
                self.optimG.step()

                progress_bar.set_postfix(loss_G=lossG.item(), loss_D=lossD.item())

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