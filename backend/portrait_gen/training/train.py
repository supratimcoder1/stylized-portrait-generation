import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from backend.portrait_gen.training.dataset import VectorFaceDataset
from backend.portrait_gen.training.model import (
    PatchGANDiscriminator,
    UNetGenerator,
    init_weights,
    weights_init_normal,
)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.network = vgg.eval().to(device)
        self.device = device

        for param in self.network.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_for_vgg(self, x):
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, generated, target):
        gen_norm = self.normalize_for_vgg(generated)
        target_norm = self.normalize_for_vgg(target)

        gen_features = self.network(gen_norm)
        target_features = self.network(target_norm)
        return self.criterion(gen_features, target_features)


def train_cgan(
    generator,
    discriminator,
    train_loader,
    val_loader,
    epochs=100,
    device="cuda",
    output_dir="training_outputs",
    checkpoint_dir="checkpoints",
):
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_L1 = nn.L1Loss().to(device)
    perceptual_criterion = PerceptualLoss(device)

    lambda_l1 = 50.0
    lambda_perceptual = 10.0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "final_model.pth")

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=25, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=25, gamma=0.5)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        with tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=140,
            bar_format="{l_bar}{bar:60}{r_bar}",
            mininterval=0.2,
        ) as pbar:
            for real_photos, stylized_targets in pbar:
                real_photos = real_photos.to(device)
                stylized_targets = stylized_targets.to(device)

                fake_targets = generator(real_photos)

                for p in discriminator.parameters():
                    p.requires_grad = True

                optimizer_D.zero_grad()

                pred_real = discriminator(real_photos, stylized_targets)
                real_labels = torch.ones_like(pred_real) * (0.8 + 0.2 * torch.rand_like(pred_real))
                loss_D_real = criterion_GAN(pred_real, real_labels)

                pred_fake = discriminator(real_photos, fake_targets.detach())
                fake_labels = torch.zeros_like(pred_fake)
                loss_D_fake = criterion_GAN(pred_fake, fake_labels)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()

                for p in discriminator.parameters():
                    p.requires_grad = False

                optimizer_G.zero_grad()

                pred_fake_for_G = discriminator(real_photos, fake_targets)
                target_real_for_G = torch.ones_like(pred_fake_for_G)
                loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)

                loss_G_L1 = criterion_L1(fake_targets, stylized_targets) * lambda_l1
                loss_G_perc = perceptual_criterion(fake_targets, stylized_targets) * lambda_perceptual

                loss_G = loss_G_GAN + loss_G_L1 + loss_G_perc
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()

                epoch_d_loss += loss_D.item()
                epoch_g_loss += loss_G.item()
                pbar.set_postfix({"D_Loss": f"{loss_D.item():.3f}", "G_Loss": f"{loss_G.item():.3f}"})

        scheduler_G.step()
        scheduler_D.step()

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_photos, val_targets = next(iter(val_loader))
            val_photos, val_targets = val_photos.to(device), val_targets.to(device)
            val_generated = generator(val_photos)

            comparison = torch.cat([val_photos, val_generated, val_targets], dim=3)
            vutils.save_image(
                comparison,
                os.path.join(output_dir, f"epoch_{epoch+1}.png"),
                normalize=True,
                value_range=(-1, 1),
                nrow=4,
            )

            for v_photos, v_targets in val_loader:
                v_photos, v_targets = v_photos.to(device), v_targets.to(device)
                v_gen = generator(v_photos)
                val_loss += criterion_L1(v_gen, v_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1} | G Loss: {epoch_g_loss / len(train_loader):.4f} "
            f"| D Loss: {epoch_d_loss / len(train_loader):.4f} | Val L1: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(generator.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (Val L1: {best_val_loss:.4f})")

    print(f"Training complete. Best model saved at: {best_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="training_outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_dataset = VectorFaceDataset("dataset/train_images", "dataset/train_targets")
    val_dataset = VectorFaceDataset("dataset/val_images", "dataset/val_targets")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    generator = UNetGenerator().to(device)
    model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Loading pretrained generator state from {model_path}")
        generator.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Pretrained generator not found, initializing weights.")
        generator.apply(init_weights)

    discriminator = PatchGANDiscriminator().to(device)
    discriminator.apply(weights_init_normal)

    train_cgan(
        generator,
        discriminator,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
