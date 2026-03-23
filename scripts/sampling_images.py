import os
import random
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "dataset" / "images"
SAMPLE_DIR = PROJECT_ROOT / "sample" / "images"

# FERET frontal variants: fa, fb, fa_a, fa_b, fb_a, fb_b
FRONTAL_PATTERN = re.compile(r"_(fa|fb|fa_a|fa_b|fb_a|fb_b)\.ppm$", re.IGNORECASE)

def create_sample_subset(count=500):
    # 1. Ensure source exists
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} not found.")
        return

    # 2. Get ONLY frontal .ppm files using strict FERET suffix matching
    all_images = [
        f.name
        for f in SOURCE_DIR.iterdir()
        if f.is_file() and FRONTAL_PATTERN.search(f.name)
    ]
    
    if len(all_images) < count:
        print(f"Warning: Only found {len(all_images)} frontal images. Sampling all of them.")
        count = len(all_images)

    # 3. Randomly select
    sampled_images = random.sample(all_images, count)

    # 4. Clear old garbage and create clean target dir
    if SAMPLE_DIR.exists():
        print("Wiping old sample batch...")
        shutil.rmtree(SAMPLE_DIR)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Copying {count} strictly frontal images to {SAMPLE_DIR}...")

    # 5. Copy files
    sampled_pose_counts = {}
    for img_name in sampled_images:
        src_path = SOURCE_DIR / img_name
        dest_path = SAMPLE_DIR / img_name
        shutil.copy2(src_path, dest_path)
        pose_key = FRONTAL_PATTERN.search(img_name).group(1).lower()
        sampled_pose_counts[pose_key] = sampled_pose_counts.get(pose_key, 0) + 1

    print("Done. You now have a mathematically safe test batch.")
    print(f"Pose distribution: {sampled_pose_counts}")

if __name__ == "__main__":
    create_sample_subset(500)