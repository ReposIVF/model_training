import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128),
}

MAX_WINDOW_SIZE: Tuple[int, int] = (1280, 720)


def setup_logger() -> None:
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def read_yolo_segmentation(label_path: Path, img_shape: Tuple[int, int]) -> List[Tuple[int, np.ndarray]]:
    """Read YOLO polygonal segmentation annotations.

    Args:
        label_path (Path): Path to YOLO annotation file.
        img_shape (Tuple[int, int]): Shape of the image (height, width).

    Returns:
        List[Tuple[int, np.ndarray]]: List of (class_id, polygon_points_in_pixels).
    """
    height, width = img_shape
    annotations: List[Tuple[int, np.ndarray]] = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 3:
                continue
            class_id = int(values[0])
            coords = np.array(values[1:], dtype=float).reshape(-1, 2)
            coords_px = np.column_stack((coords[:, 0] * width, coords[:, 1] * height)).astype(int)
            annotations.append((class_id, coords_px))

    return annotations


def resize_image_and_polygons(
    image: np.ndarray, annotations: List[Tuple[int, np.ndarray]], max_size: Tuple[int, int]
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """Resize image and polygons to fit within a maximum display size.

    Args:
        image (np.ndarray): Input image.
        annotations (List[Tuple[int, np.ndarray]]): YOLO polygon annotations.
        max_size (Tuple[int, int]): Max (width, height) of display window.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]: Resized image and scaled annotations.
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        annotations_rescaled = [
            (cls_id, (polygon * scale).astype(int)) for cls_id, polygon in annotations
        ]
        return image_resized, annotations_rescaled
    return image, annotations


def draw_polygons(image: np.ndarray, annotations: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    """Draw YOLO segmentation polygons on the image.

    Args:
        image (np.ndarray): Image to draw on.
        annotations (List[Tuple[int, np.ndarray]]): List of class and polygon points.

    Returns:
        np.ndarray: Annotated image.
    """
    for class_id, polygon in annotations:
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], color)
        image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    return image


def visualize_images(
    images_dir: Path,
    labels_dir: Path,
    window_name: str = "YOLO Segmentation Viewer",
) -> None:
    """Visualize YOLO segmentation annotations over images.

    Args:
        images_dir (Path): Directory containing .png images.
        labels_dir (Path): Directory containing YOLO label .txt files.
        window_name (str): Name of the display window.
    """
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        logging.error("No .png images found in %s", images_dir)
        return

    for idx, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Could not read image: %s", image_path)
            continue

        annotations: List[Tuple[int, np.ndarray]] = []
        if label_path.exists():
            annotations = read_yolo_segmentation(label_path, image.shape[:2])
        else:
            logging.warning("Annotation not found for image: %s", image_path.name)

        image, annotations = resize_image_and_polygons(image, annotations, MAX_WINDOW_SIZE)
        image = draw_polygons(image, annotations)

        cv2.imshow(window_name, image)
        logging.info("Showing image %d / %d: %s", idx, len(image_paths), image_path.name)
        key = cv2.waitKey(0)

        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()


def main() -> None:
    """Main entry point."""
    setup_logger()
    images_dir = Path(input("Enter path to images folder: ").strip())
    labels_dir = Path(input("Enter path to labels folder: ").strip())

    if not images_dir.exists() or not labels_dir.exists():
        logging.error("Invalid input paths.")
        return

    visualize_images(images_dir, labels_dir)


if __name__ == "__main__":
    main()
