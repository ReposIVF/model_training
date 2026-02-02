"""
@description: Crop embryos from images using a YOLO model.
"""

import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from utils.model_explorer import get_cropper_model_weights


def cropper(embryos_list: list, models: list) -> list:
    """
    Crop embryos from images using a YOLO model.

    Args:
        embryos_list (list): List of embryo dicts. Each must contain 'image' path.
        models (list): List of model dicts with 'id', 'type', and 'filename'.

    Returns:
        list: Same embryos list, with possibly updated 'isEmbryo' and overwritten cropped images.
    """
    weights_path = get_cropper_model_weights(models)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] Cropper model not found at: {weights_path}")

    try:
        model = YOLO(weights_path)
        model.to('cpu')
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load YOLO model: {e}")

    conf_thresh = 0.80

    for embryo in embryos_list:
        image_path = embryo.get('image')
        embryo['isEmbryo'] = False  # default

        if not image_path or not os.path.exists(image_path):
            print(f"[WARNING] Invalid or missing image path: {image_path}")
            continue

        try:
            print(f"[INFO] Cropping embryo from image: {image_path}")
            img_cv2 = cv2.imread(image_path)
            if img_cv2 is None:
                print(f"[WARNING] Cannot read image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            results = model.predict(img_rgb, conf=conf_thresh, device='cpu')

            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy()
            embryo_boxes = [box for box, cls in zip(boxes, class_ids) if cls == 0]

            if not embryo_boxes:
                print(f"[INFO] No embryos detected in {image_path}")
                continue

            print(f"[INFO] Detected {len(embryo_boxes)} embryo(s) in {image_path}")
            embryo['isEmbryo'] = True

            # Use only the first detection box
            box = embryo_boxes[0]
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

            cropped_img = Image.fromarray(img_rgb).crop((xmin, ymin, xmax, ymax))
            cropped_img.save(image_path)  # Overwrite original

        except Exception as e:
            print(f"[ERROR] Error processing {image_path}: {e}")
            continue

    return embryos_list