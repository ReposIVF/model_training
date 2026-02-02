# inference/utils.py

from ultralytics import YOLO
from typing import Optional


def load_model(model_path: str = "SpermSegmentation/best.pt") -> YOLO:
    """Load a YOLOv8 segmentation model.

    Args:
        model_path (str): Path to the .pt model file.

    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_path)
    return model
