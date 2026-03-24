import random
import shutil
from pathlib import Path

SAMPLE_IMAGES = Path("sample/images")
SAMPLE_TARGETS = Path("sample/targets")
DATASET_DIR = Path("sample_dataset")

def split_dataset(train_ratio=0.9):
    train_images_dir = DATASET_DIR / "train_images"
    train_targets_dir = DATASET_DIR / "train_targets"
    val_images_dir = DATASET_DIR / "val_images"
    val_targets_dir = DATASET_DIR / "val_targets"

    for d in [train_images_dir, train_targets_dir, val_images_dir, val_targets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([f.name for f in SAMPLE_IMAGES.iterdir() if f.is_file()])
    
    # Ensure targets exist for these images (targets are saved as .png in our previous script)
    valid_pairs = []
    for img in images:
        target_name = Path(img).stem + ".png"
        if (SAMPLE_TARGETS / target_name).exists():
            valid_pairs.append((img, target_name))

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Moving {len(train_pairs)} to train, {len(val_pairs)} to val...")

    for img_name, target_name in train_pairs:
        shutil.copy2(SAMPLE_IMAGES / img_name, train_images_dir / img_name)
        shutil.copy2(SAMPLE_TARGETS / target_name, train_targets_dir / target_name)

    for img_name, target_name in val_pairs:
        shutil.copy2(SAMPLE_IMAGES / img_name, val_images_dir / img_name)
        shutil.copy2(SAMPLE_TARGETS / target_name, val_targets_dir / target_name)

    print("Splits created successfully.")

if __name__ == "__main__":
    split_dataset()
