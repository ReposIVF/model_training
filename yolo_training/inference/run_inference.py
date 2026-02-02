# inference/inference.py

import cv2
import os
import logging
from typing import Optional
from inference.utils import load_model
from inference.annotator import draw_detections
from collections import Counter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_inference_on_video(
    video_path: str,
    output_path: str = "annotated_output.mp4",
    model_path: str = "SpermSegmentation/best.pt",
    threshold: float = 0.5
) -> None:
    """Run inference on a video using a YOLOv8 segmentation model.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the annotated video.
        model_path (str): Path to the trained YOLO model.
        threshold (float): Detection confidence threshold.
    """
    model = load_model(model_path)
    logger.info(f"Loaded model from: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {total_frames}, Resolution: {width}x{height}, FPS: {fps}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=threshold, verbose=False)[0]

        boxes = results.boxes.xyxy.int().tolist() if results.boxes else []
        masks = results.masks.data.cpu().numpy() if results.masks else []
        class_ids = results.boxes.cls.int().tolist() if results.boxes else []
        scores = results.boxes.conf.tolist() if results.boxes else []

        # Resize masks to match frame size
        processed_masks = [cv2.resize(mask, (frame.shape[1], frame.shape[0])) for mask in masks]

        annotated_frame = draw_detections(frame, boxes, processed_masks, class_ids, scores, threshold)
        out.write(annotated_frame)

        # Logging por frame
        class_counter = Counter(class_ids)
        summary = ", ".join(f"{cls}: {cnt}" for cls, cnt in sorted(class_counter.items()))
        logger.info(f"Frame {frame_idx+1}/{total_frames} - Objects detected: {summary if summary else 'None'}")

        frame_idx += 1

    cap.release()
    out.release()
    logger.info(f"Inference completed. Output saved to: {output_path}")
