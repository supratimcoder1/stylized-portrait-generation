import argparse
from pathlib import Path
from typing import Optional, Union

from backend.portrait_gen.inference.service import StylizeParams, VectorPortraitService


class VectorPortraitPipeline:
    def __init__(self, weights_path: Union[str, Path], device: str = "cpu"):
        self.service = VectorPortraitService(weights_path=weights_path, device=device)

    def generate(
        self,
        input_path: str,
        output_path: str,
        restore_size: bool = True,
        style_params: Optional[StylizeParams] = None,
    ) -> None:
        self.service.generate_to_path(
            input_path=input_path,
            output_path=output_path,
            restore_size=restore_size,
            params=style_params,
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Generate a vector portrait from an input image.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to save output image")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=str(project_root / "checkpoints" / "final_model.pth"),
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


if __name__ == "__main__":
    main()
