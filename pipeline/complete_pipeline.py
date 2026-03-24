import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Dynamically add the project root to sys.path so we can import from training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.model import UNetGenerator


class VectorPortraitPipeline:
    def __init__(self, weights_path: Union[str, Path], device: str = "cpu"):
        self.device = torch.device(device)
        self.model = UNetGenerator().to(self.device)

        # Map weights to the selected target device (CPU by default)
        state_dict = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.image_size = 256

    def preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        # Match dataset preprocessing used during training
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize to [-1, 1] to match Tanh-based generator output space
        image_float = (image.astype(np.float32) / 127.5) - 1.0

        # HWC -> CHW and add batch dimension
        tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).detach().cpu()

        # CHW -> HWC
        image_float = tensor.numpy().transpose(1, 2, 0)

        # Un-normalize from [-1, 1] to [0, 255]
        image_unnorm = (image_float + 1.0) * 127.5
        image_uint8 = np.clip(image_unnorm, 0, 255).astype(np.uint8)

        # Convert to OpenCV BGR for writing
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        if original_size:
            # Slightly smoother upscale when restoring original resolution
            image_bgr = cv2.resize(image_bgr, original_size, interpolation=cv2.INTER_CUBIC)

        return image_bgr

    def generate(self, input_path: str, output_path: str, restore_size: bool = True) -> None:
        original_image = cv2.imread(input_path)
        if original_image is None:
            raise ValueError("Invalid input image.")

        orig_w, orig_h = original_image.shape[1], original_image.shape[0]

        input_tensor = self.preprocess(input_path)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        target_size = (orig_w, orig_h) if restore_size else None
        final_image = self.postprocess(output_tensor, target_size)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path_obj), final_image):
            raise RuntimeError(f"Failed to save output image to {output_path_obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a vector portrait from an input image.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to save output image")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "final_model.pth"),
        help="Path to UNet weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference (for example: cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--keep-size",
        action="store_true",
        help="Keep output at 256x256 instead of original size",
    )

    args = parser.parse_args()

    try:
        pipeline = VectorPortraitPipeline(weights_path=args.weights, device=args.device)
        pipeline.generate(args.input, args.output, restore_size=not args.keep_size)
        print(f"Success. Vector portrait saved to {args.output}")
    except Exception as exc:
        print(f"Inference failed: {exc}")
        raise SystemExit(1)
