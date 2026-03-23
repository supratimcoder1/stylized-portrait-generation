import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Assuming you have your dataset and U-Net classes defined
# from dataset import VectorFaceDataset
# from model import UNetGenerator

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pretrained VGG16 and extract features up to relu3_3
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.network = vgg.eval().to(device)
        self.device = device
        
        # Freeze VGG weights
        for param in self.network.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
        # ImageNet Normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_for_vgg(self, x):
        # First, convert from Tanh [-1, 1] range back to [0, 1]
        x_01 = (x + 1.0) / 2.0
        # Then apply ImageNet normalization
        return (x_01 - self.mean) / self.std

    def forward(self, generated, target):
        gen_norm = self.normalize_for_vgg(generated)
        target_norm = self.normalize_for_vgg(target)
        
        gen_features = self.network(gen_norm)
        target_features = self.network(target_norm)
        return self.criterion(gen_features, target_features)

def train_robust_baseline(generator, train_loader, val_loader, epochs=100, device='cuda', output_dir='training_outputs', checkpoint_dir='checkpoints'):
    generator = generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    l1_criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss(device)
    
    # Aggressive loss weights for structure
    lambda_l1 = 75.0
    lambda_perceptual = 10.0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
            for real_photos, stylized_targets in pbar:
                real_photos = real_photos.to(device)
                stylized_targets = stylized_targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                generated_outputs = generator(real_photos)
                
                # Calculate losses
                loss_l1 = l1_criterion(generated_outputs, stylized_targets) * lambda_l1
                loss_perc = perceptual_criterion(generated_outputs, stylized_targets) * lambda_perceptual
                total_loss = loss_l1 + loss_perc
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                pbar.set_postfix({'L1': f"{loss_l1.item():.2f}", 'Perc': f"{loss_perc.item():.2f}"})
                
        # --- Validation & Visual Logging ---
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Grab just one batch for visual logging
            val_photos, val_targets = next(iter(val_loader))
            val_photos, val_targets = val_photos.to(device), val_targets.to(device)
            val_generated = generator(val_photos)
            
            # Stack Real | Generated | Target side-by-side
            comparison = torch.cat([val_photos, val_generated, val_targets], dim=3)
            # value_range=(-1, 1) properly handles the Tanh outputs
            vutils.save_image(
                comparison, 
                os.path.join(output_dir, f"epoch_{epoch+1}.png"), 
                normalize=True, 
                value_range=(-1, 1), 
                nrow=4
            )
            
            # Calculate basic validation loss
            for v_photos, v_targets in val_loader:
                v_photos, v_targets = v_photos.to(device), v_targets.to(device)
                v_gen = generator(v_photos)
                val_loss += l1_criterion(v_gen, v_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val L1: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(generator.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (Val L1: {best_val_loss:.4f})")

    print(f"Training complete. Best model saved at: {best_model_path}")

# Execution setup example:
# train_dataset = VectorFaceDataset("dataset/train_images", "dataset/train_targets")
# val_dataset = VectorFaceDataset("dataset/val_images", "dataset/val_targets")
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# generator = UNetGenerator()
# train_robust_baseline(generator, train_loader, val_loader)

if __name__ == '__main__':
    import argparse
    import os
    from dataset import VectorFaceDataset
    from model import UNetGenerator, init_weights
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='training_outputs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    
    train_dataset = VectorFaceDataset('dataset/train_images', 'dataset/train_targets')
    val_dataset = VectorFaceDataset('dataset/val_images', 'dataset/val_targets')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    generator = UNetGenerator().to(device)
    generator.apply(init_weights)
    
    train_robust_baseline(generator, train_loader, val_loader, epochs=args.epochs, device=device, output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir)

