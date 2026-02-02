# inference/annotator.py

import cv2
import numpy as np
from typing import List, Tuple


# Clase -> color BGR
CLASS_COLORS = {
    0: (255, 0, 0),     # sperm - azul
    1: (0, 255, 0),     # needle - verde
    2: (0, 0, 255),     # defect - rojo
    3: (255, 255, 0),   # other - celeste
    4: (255, 0, 255),   # egg - magenta
    5: (0, 255, 255),   # pipette - amarillo
    6: (128, 0, 128)        # ntail - morado
}

CLASS_NAMES = {
    0: "sperm",
    1: "needle",
    2: "defect",
    3: "other",
    4: "egg",
    5: "pipette",
    6: "ntail"
}


def draw_detections(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    masks: List[np.ndarray],
    class_ids: List[int],
    scores: List[float],
    threshold: float = 0.5
) -> np.ndarray:
    """Draw bounding boxes, class labels, and masks on the frame.

    Args:
        frame (np.ndarray): Original video frame.
        boxes (List[Tuple[int, int, int, int]]): Bounding boxes (x1, y1, x2, y2).
        masks (List[np.ndarray]): List of boolean masks (same size as frame).
        class_ids (List[int]): Class IDs corresponding to the detections.
        scores (List[float]): Confidence scores for each detection.
        threshold (float): Detection threshold to filter low-confidence outputs.

    Returns:
        np.ndarray: Frame with annotations.
    """
    overlay = frame.copy()

    for box, mask, class_id, score in zip(boxes, masks, class_ids, scores):
        if score < threshold:
            continue

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        label = CLASS_NAMES.get(class_id, str(class_id))

        # Draw mask
        if mask is not None:
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0.5] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

        # Draw bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(
            overlay,
            f"{label}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return overlay
