import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class VectorFaceDataset(Dataset):
    def __init__(self, images_dir: str, targets_dir: str, image_size: int = 256):
        self.images_dir = Path(images_dir)
        self.targets_dir = Path(targets_dir)
        self.image_size = image_size
        
        # Strictly filter for files that have a valid stylized target
        self.image_filenames = [
            f.name for f in self.images_dir.iterdir() 
            if f.is_file() and (self.targets_dir / (f.stem + ".png")).exists()
        ]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        target_name = Path(img_name).stem + ".png"
        
        img_path = str(self.images_dir / img_name)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        target_path = str(self.targets_dir / target_name)
        target = cv2.imread(target_path)

        if image is None or target is None:
            raise RuntimeError(f"Failed reading image pair for {img_name}. Check file integrity.")

        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        image = (image.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        image = torch.from_numpy(image.transpose(2, 0, 1))
        target = torch.from_numpy(target.transpose(2, 0, 1))
        
        return image, target