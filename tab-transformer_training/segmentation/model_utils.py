import os
from typing import Optional
from ultralytics import YOLO


def load_yolo_segmentation_model(weights_path: str, device: str) -> Optional[YOLO]:
    """
    Load the YOLOv8 segmentation model.

    Args:
        weights_path (str): Path to the YOLOv8 weights file.
        device (str): Device to load the model ('cpu' or 'cuda').

    Returns:
        Optional[YOLO]: Loaded YOLOv8 model, or None if loading fails.

    Raises:
        FileNotFoundError: If the weights file is not found.
    """
    if not os.path.isfile(weights_path):
        print(f"Model weights file not found: {weights_path}")
        return None

    try:
        model = YOLO(weights_path)
        model.to(device)
        print(f"Model loaded successfully from {weights_path} on {device}.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
