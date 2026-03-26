from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "backend" / "tools" / "data_prep" / "sampling_images.py"
    run_path(str(target), run_name="__main__")
