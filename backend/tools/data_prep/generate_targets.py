"""Generate structure-aware vector targets using MediaPipe and Mean Shift."""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    mp = None

DEFAULT_INPUT_DIR = Path("filtered_dataset/images")
DEFAULT_OUTPUT_DIR = Path("filtered_dataset/targets")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}

# MediaPipe landmark indices for high-detail features
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
HAAR_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def create_feature_mask(image_shape, face_landmarks) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    features = [LEFT_EYE, RIGHT_EYE, LIPS, LEFT_EYEBROW, RIGHT_EYEBROW]
    
    for feature_indices in features:
        points = np.array([
            [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
            for i in feature_indices
        ], dtype=np.int32)
        
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
    return mask

def create_haar_face_mask(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = HAAR_FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    center = (x + w // 2, y + h // 2)
    axes = (max(1, int(w * 0.42)), max(1, int(h * 0.52)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def build_feature_mask(image: np.ndarray) -> np.ndarray | None:
    if mp is not None:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
            ) as face_mesh:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    return create_feature_mask(image.shape, results.multi_face_landmarks[0])
        except Exception:
            pass

    return create_haar_face_mask(image)

def normalize_image_for_processing(image: np.ndarray) -> np.ndarray | None:
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3:
        return None

    channels = image.shape[2]
    if channels == 3:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return None

def process_single_image(args_tuple) -> tuple[bool, str]:
    image_path, output_dir, num_colors, sp, sr = args_tuple
    output_path = output_dir / f"{image_path.stem}.png"
    
    if output_path.exists():
        return True, "exists"
        
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return False, f"Failed to read image: {image_path}"

    image = normalize_image_for_processing(image)
    if image is None:
        return False, f"Unsupported image shape/channels for {image_path}"

    try:
        mask_8u = build_feature_mask(image)
        if mask_8u is None:
            mask_8u = np.zeros(image.shape[:2], dtype=np.uint8)

        mask_blurred = cv2.GaussianBlur(mask_8u, (15, 15), 0)
        mask_float = np.expand_dims(mask_blurred.astype(np.float32) / 255.0, axis=2)

        # 1. Base Passes: Heavy blur for skin, light blur for features
        flat_pass = cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)
        sharp_pass = cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)

        # 2. Composite: Blend the continuous-color images first to preserve feature boundaries
        blended = (sharp_pass * mask_float) + (flat_pass * (1.0 - mask_float))
        blended = blended.astype(np.uint8)

        # 3. Global Vectorization: Force the entire blended image into a single strict color palette
        pixels = blended.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
        )
        quantized = centers[labels.flatten()].reshape(blended.shape).astype(np.uint8)
        
        # 4. Final Polish: Light cleanup of K-Means artifacts (size 5 keeps borders sharp)
        final_output = cv2.medianBlur(quantized, 5)
        
        if not cv2.imwrite(str(output_path), final_output):
            return False, f"Failed to write target: {output_path}"
    except Exception as exc:
        return False, f"Processing failed for {image_path.name}: {exc}"
    
    return True, "generated"

def prepare_dataset_parallel(input_dir: Path, output_dir: Path, num_colors: int, sp: int, sr: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _list_images(input_dir)
    
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    tasks = [(path, output_dir, num_colors, sp, sr) for path in image_paths]
    max_workers = max(1, os.cpu_count() - 2) 
    
    print(f"Processing {len(tasks)} images into {output_dir}...")

    completed = 0
    failed_messages = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Hybrid Targets"):
            try:
                ok, message = future.result()
            except Exception as exc:
                failed_messages.append(str(exc))
                continue

            if ok:
                completed += 1
            else:
                failed_messages.append(message)

    print(f"Finished processing {completed}/{len(tasks)} images.")
    if failed_messages:
        print(f"Encountered {len(failed_messages)} failures. Sample failures:")
        for message in failed_messages[:10]:
            print(f"- {message}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare structure-aware vector targets")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--colors", type=int, default=8, help="Number of final palette colors")
    parser.add_argument("--sp", type=int, default=30, help="Spatial radius for flat skin")
    parser.add_argument("--sr", type=int, default=50, help="Color radius for flat skin")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare_dataset_parallel(args.input_dir, args.output_dir, args.colors, args.sp, args.sr)