import os
from typing import List
import cv2
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO


def draw_bounding_boxes(image: Image.Image, embryos: List, output_path: str) -> None:
    """
    Draw bounding boxes on the image and save it.

    Args:
        image (PIL.Image): The input image.
        embryos (list): Detected bounding boxes.
        output_path (str): Output path for the image with boxes.
    """
    draw = ImageDraw.Draw(image)
    for box in embryos:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
    image.save(output_path)


def crop_embryos(
    model: YOLO,
    image_path: str,
    output_dir: str,
    device: str,
    conf_thresh: float
) -> List[np.ndarray]:
    """
    Detect and crop embryos from an image.

    Args:
        model (YOLO): YOLOv8 model.
        image_path (str): Path to the input image.
        output_dir (str): Output directory for results.
        device (str): Device for inference ('cpu' or 'cuda').
        conf_thresh (float): Confidence threshold.

    Returns:
        List[np.ndarray]: List of cropped embryo images as numpy arrays.
    """
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return None
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, conf=conf_thresh, device=device)

    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy()
    embryos = [box for box, cls in zip(boxes, class_ids) if cls == 0]

    if not embryos:
        print(f"No embryos found in image: {image_path}")
        return []

    img_pil = Image.fromarray(img_rgb)
    boxed_dir = os.path.join(output_dir, "boxed_images")
    cropped_dir = os.path.join(output_dir, "cropped_embryos")
    os.makedirs(boxed_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    boxed_image_path = os.path.join(boxed_dir, f"boxed_{os.path.basename(image_path)}")
    draw_bounding_boxes(img_pil.copy(), embryos, boxed_image_path)

    cropped_images = []
    for index, box in enumerate(embryos):
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        cropped_img = img_pil.crop((xmin, ymin, xmax, ymax))
        cropped_images.append(np.array(cropped_img))

        #cropped_image_path = os.path.join(cropped_dir, f"cropped_{index}_{os.path.basename(image_path)}")
        cropped_image_path = os.path.join(cropped_dir, f"{os.path.basename(image_path)}")
        cropped_img.save(cropped_image_path)

    return cropped_images


def process_images_in_folder(folder_path: str, model: YOLO, output_dir: str, device: str, conf_thresh: float) -> List:
    """
    Process all images in a folder to detect and crop embryos.

    Args:
        folder_path (str): Path to folder with images.
        model: YOLOv8 model.
        output_dir (str): Directory to save results.
        device (str): Device for inference.
        conf_thresh (float): Confidence threshold.
    """
    supported_ext = (".png", ".jpg", ".jpeg", ".bmp")
    cropped_images = []

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(supported_ext):
                image_path = os.path.join(dirpath, filename)
                cropped = crop_embryos(model, image_path, output_dir, device, conf_thresh)
                if cropped:
                    if isinstance(cropped, list):
                        cropped_images.extend(cropped)
                    else:
                        cropped_images.append(cropped)

    return cropped_images
