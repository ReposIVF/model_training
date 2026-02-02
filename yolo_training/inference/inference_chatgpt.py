#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128),
}


def setup_logger() -> None:
    """Configure logging format and level."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="YOLO segmentation inference on video.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video.")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency factor for overlay [0-1].")
    parser.add_argument("--show", type=bool, default=False, help="Show video in real-time during inference.")
    return parser.parse_args()


def overlay_masks(frame: np.ndarray, masks: np.ndarray, classes: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay segmentation masks on frame with transparency.

    Args:
        frame (np.ndarray): Original image frame (BGR).
        masks (np.ndarray): Segmentation masks array (N x H x W).
        classes (np.ndarray): Class indices for each mask.
        alpha (float): Transparency factor [0-1].

    Returns:
        np.ndarray: Frame with overlay applied.
    """
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    for idx, mask in enumerate(masks):
        color = CLASS_COLORS.get(int(classes[idx]), (255, 255, 255))
        mask_resized = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask_resized.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

    return overlay


def run_inference(model_path: str, video_path: str, output_path: str, device: str, alpha: float, show: bool) -> None:
    """Run YOLO segmentation inference on a video file.

    Args:
        model_path (str): Path to YOLO model file.
        video_path (str): Path to input video file.
        output_path (str): Path to output video file.
        device (str): Device to use (cuda or cpu).
        alpha (float): Transparency factor.
        show (bool): Whether to display video in real-time.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    logging.info(f"Processing video: {video_path}")
    logging.info(f"Saving output to: {output_path}")

    for _ in tqdm(range(total_frames), desc="Inference Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device=device, verbose=False)
        result = results[0]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            frame = overlay_masks(frame, masks, classes, alpha)

        writer.write(frame)

        if show:
            cv2.imshow("YOLO Segmentation Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Real-time visualization stopped by user.")
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logging.info("Inference completed successfully.")


def main() -> None:
    """Main entry point."""
    setup_logger()
    args = parse_arguments()
    run_inference(args.model, args.source, args.output, args.device, args.alpha, args.show)


if __name__ == "__main__":
    main()
