import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "dataset" / "images"
TARGET_DIR = PROJECT_ROOT / "filtered_dataset" / "images"

# FERET frontal variants: fa, fb, fa_a, fa_b, fa_c, fb_a, fb_b, fb_c
FRONTAL_PATTERN = re.compile(r"_(fa|fb|fa_a|fa_b|fa_c|fb_a|fb_b|fb_c)\.ppm$", re.IGNORECASE)


def filter_frontal_images(clear_target: bool = True) -> None:
    # 1. Ensure source exists
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} not found.")
        return

    # 2. Collect only frontal .ppm files
    frontal_images = [
        f.name
        for f in SOURCE_DIR.iterdir()
        if f.is_file() and FRONTAL_PATTERN.search(f.name)
    ]

    if not frontal_images:
        print("No frontal images found in source directory.")
        return

    # 3. Reset output directory for a clean run
    if clear_target and TARGET_DIR.exists():
        print("Wiping old filtered dataset...")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Copy filtered files and track pose counts
    pose_counts = {}
    for img_name in frontal_images:
        src_path = SOURCE_DIR / img_name
        dst_path = TARGET_DIR / img_name
        shutil.copy2(src_path, dst_path)

        pose_key = FRONTAL_PATTERN.search(img_name).group(1).lower()
        pose_counts[pose_key] = pose_counts.get(pose_key, 0) + 1

    print(f"Copied {len(frontal_images)} frontal images to {TARGET_DIR}.")
    print(f"Pose distribution: {pose_counts}")


if __name__ == "__main__":
    filter_frontal_images(clear_target=True)
