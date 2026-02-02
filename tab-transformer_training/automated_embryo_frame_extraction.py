import os
import cv2
import logging
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

def get_video_files(input_folder: Path, extension: str = ".mp4") -> List[Path]:
    return [file for file in input_folder.glob(f"*{extension}") if file.is_file()]

def create_output_folder(base_output_folder: Path, video_name: str) -> Path:
    output_folder = base_output_folder / video_name
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

def load_yolo_model(model_path: str = "yolov8n.pt", device: str = "cuda") -> YOLO:
    model = YOLO(model_path)
    model.to(device)
    return model

def format_minute_second(frame_idx: int, fps: float) -> str:
    frame_time = frame_idx / fps
    minutes = int(frame_time // 60)
    seconds = int(frame_time % 60)
    return f"{minutes:02d}-{seconds:02d}"

def crop_and_save(image, bbox: Tuple[int, int, int, int], output_path: Path) -> None:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), crop)

def process_video(
    video_path: Path,
    output_base_folder: Path,
    model: YOLO,
    confidence_threshold: float = 0.3,
) -> None:
    video_name = video_path.stem
    output_folder = create_output_folder(output_base_folder, video_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    crop_id = 1
    last_saved_minute_second = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        minute_second = format_minute_second(frame_idx, fps)

        if minute_second != last_saved_minute_second:
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False,
                device=0 if model.device.type == "cuda" else "cpu",
            )

            if len(results[0].boxes) > 0:
                detection = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = detection[:4]
                filename = f"{video_name}_{minute_second}_id_{crop_id}.png"
                output_path = output_folder / filename
                crop_and_save(frame, (x1, y1, x2, y2), output_path)
                crop_id += 1
                last_saved_minute_second = minute_second

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    logging.info(f"Completed processing: {video_name}")

def process_videos_in_folder(
    input_folder: Path,
    output_folder: Path,
    model_path: str = "yolov8n.pt",
    device: str = "cuda",
    confidence_threshold: float = 0.3,
) -> None:
    model = load_yolo_model(model_path, device)
    video_files = get_video_files(input_folder)

    for video_file in video_files:
        process_video(
            video_path=video_file,
            output_base_folder=output_folder,
            model=model,
            confidence_threshold=confidence_threshold,
        )

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Process videos with YOLOv8 and save unique crops per second")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save detection crops")
    parser.add_argument("--model_path", type=str, default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="Confidence threshold for detections")

    args = parser.parse_args()

    process_videos_in_folder(
        input_folder=Path(args.input_dir),
        output_folder=Path(args.output_dir),
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
