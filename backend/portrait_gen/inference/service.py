from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from backend.portrait_gen.training.model import UNetGenerator


@dataclass(frozen=True)
class StylizeParams:
    num_colors: int = 8
    edge_weight: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0


class VectorPortraitService:
    def __init__(self, weights_path: Union[str, Path], device: str = "cpu"):
        self.device = torch.device(device)
        self.model = UNetGenerator().to(self.device)

        state_dict = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.image_size = 256

    def apply_style_sliders(self, image: np.ndarray, params: StylizeParams) -> np.ndarray:
        if params.contrast != 1.0:
            image = cv2.convertScaleAbs(image, alpha=params.contrast, beta=0)

        if params.saturation != 1.0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.saturation, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if params.num_colors < 256:
            step = 256 / params.num_colors
            indices = np.floor(image / step)
            image = (indices * step).astype(np.uint8)

        if params.edge_weight > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
            )
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            image = cv2.addWeighted(image, 1.0, edges_bgr, -params.edge_weight, 0)
            image[edges == 0] = [0, 0, 0]

        return image

    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> torch.Tensor:
        if isinstance(image_input, np.ndarray):
            image = image_input.copy()
            image_source = "provided array"
        else:
            image = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
            image_source = str(image_input)

        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_source}")

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

        image_float = (image.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        tensor = tensor.squeeze(0).detach().cpu()
        image_float = tensor.numpy().transpose(1, 2, 0)
        image_unnorm = (image_float + 1.0) * 127.5
        image_uint8 = np.clip(image_unnorm, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        if original_size:
            image_bgr = cv2.resize(image_bgr, original_size, interpolation=cv2.INTER_CUBIC)

        return image_bgr

    def generate_from_array(
        self,
        image: np.ndarray,
        restore_size: bool = True,
        params: Optional[StylizeParams] = None,
    ) -> np.ndarray:
        if image is None:
            raise ValueError("Invalid input image.")

        orig_w, orig_h = image.shape[1], image.shape[0]
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        target_size = (orig_w, orig_h) if restore_size else None
        output_image = self.postprocess(output_tensor, target_size)

        if params is not None:
            output_image = self.apply_style_sliders(output_image, params)

        return output_image

    def generate_to_path(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        restore_size: bool = True,
        params: Optional[StylizeParams] = None,
    ) -> Path:
        original_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        output_image = self.generate_from_array(
            original_image,
            restore_size=restore_size,
            params=params,
        )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path_obj), output_image):
            raise RuntimeError(f"Failed to save output image to {output_path_obj}")

        return output_path_obj
