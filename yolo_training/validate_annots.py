import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def get_image_extensions() -> Tuple[str, ...]:
    return ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"


def validate_class_id(class_id: int, num_classes: int) -> bool:
    return 0 <= class_id < num_classes


def validate_polygon(points: List[float]) -> bool:
    if len(points) < 6:
        return False
    if len(points) % 2 != 0:
        return False
    return all(0.0 <= value <= 1.0 for value in points)


def polygon_to_bbox(points: List[float]) -> Tuple[float, float, float, float]:
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    return x_min, y_min, width, height


def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    _, _, width, height = bbox
    return width > 0.0 and height > 0.0


def load_image_shape(image_path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Image cannot be read")
    height, width = image.shape[:2]
    return height, width


def validate_label_file(
    label_path: Path,
    image_path: Path,
    num_classes: int,
) -> List[str]:
    errors: List[str] = []

    if not label_path.exists():
        errors.append("Missing label file")
        return errors

    if not image_path.exists():
        errors.append("Missing image file")
        return errors

    try:
        load_image_shape(image_path)
    except ValueError:
        errors.append("Unreadable image")
        return errors

    content = label_path.read_text().strip()
    if not content:
        errors.append("Empty label file")
        return errors

    for line_idx, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 7:
            errors.append(f"Line {line_idx}: insufficient values")
            continue

        try:
            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
        except ValueError:
            errors.append(f"Line {line_idx}: non-numeric values")
            continue

        if not validate_class_id(class_id, num_classes):
            errors.append(f"Line {line_idx}: invalid class id {class_id}")

        if not validate_polygon(points):
            errors.append(f"Line {line_idx}: invalid polygon")

        bbox = polygon_to_bbox(points)
        if not validate_bbox(bbox):
            errors.append(f"Line {line_idx}: invalid derived bbox")

    return errors





def find_corresponding_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in get_image_extensions():
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None



def validate_dataset(dataset_path: Path, num_classes: int) -> None:
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logging.warning(f"Skipping {dataset_path.name}: missing images or labels folder")
        return

    for label_file in labels_dir.glob("*.txt"):
        image_path = find_corresponding_image(images_dir, label_file.stem)

        errors = validate_label_file(
            label_path=label_file,
            image_path=image_path if image_path else Path(""),
            num_classes=num_classes,
        )

        if errors:
            for error in errors:
                logging.error(f"{dataset_path.name}/{label_file.name}: {error}")


def validate_all_datasets(root_dir: Path, num_classes: int) -> None:
    for dataset_dir in root_dir.iterdir():
        if dataset_dir.is_dir():
            logging.info(f"Validating dataset: {dataset_dir.name}")
            validate_dataset(dataset_dir, num_classes)


def main() -> None:
    configure_logging()

    datasets_root = Path("datasets_tail")
    num_classes = 7

    validate_all_datasets(datasets_root, num_classes)


if __name__ == "__main__":
    main()
