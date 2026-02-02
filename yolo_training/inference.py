from inference.run_inference import run_inference_on_video

test = run_inference_on_video(
    video_path="tmp/input/SrP54v9IEd-1070-2_2025-09-15-06-25-Camera.mp4",
    output_path="tmp/output/SrP54v9IEd-1070-2_2025-09-15-06-25-Camera_out_old_model_without_tail.mp4",
    model_path="models/all_datasets_exp1.pt",
    threshold=0.4
)