import bz2
import shutil
from pathlib import Path
from tqdm import tqdm

FERET_ROOT = Path("./colorferet")
OUTPUT_DIR = Path("./dataset/images")

def flatten_feret_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Explicitly target only the high-res images inside data/images/
    print("Scanning for high-res images...")
    archive_paths = list(FERET_ROOT.rglob("**/data/images/**/*.ppm.bz2"))
    
    if not archive_paths:
        print("No high-res images found. Check your FERET_ROOT path.")
        return

    print(f"Found {len(archive_paths)} high-res images. Decompressing...")
    
    for bz2_path in tqdm(archive_paths):
        # Drop the .bz2 extension
        output_path = OUTPUT_DIR / bz2_path.stem 
        
        if output_path.exists():
            continue
            
        try:
            with bz2.BZ2File(bz2_path, 'rb') as source, open(output_path, 'wb') as dest:
                shutil.copyfileobj(source, dest)
        except Exception as e:
            print(f"Failed to decompress {bz2_path.name}: {e}")

    print(f"Extraction complete. Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    flatten_feret_dataset()