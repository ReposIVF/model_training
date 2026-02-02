from ultralytics import YOLO
import os
import mlflow

# Set the environment variables for MLflow, adjust MLFLOW_TRACKING_URI depends on your repo location
os.environ['MLFLOW_TRACKING_URI'] = 'file:///C:/Users/Usuario/VSCodeProjects/yolo_ultralytics_segmentation_training/mlruns'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'YOLOv5-seg sperm_dataset'  # Replace with your experiment name
os.environ['MLFLOW_RUN'] = 'YOLOv5-seg sperm_dataset'  # Replace with your run name

model = YOLO('yolov5s-seg.pt')

if __name__ == '__main__':
    params = {
        "data": 'datasets/tail.yaml',
        "epochs": 60,
        "imgsz": 640,
        "workers": 4,
        "batch": 16
    }
    mlflow.set_tag('mlflow.runName', 'YOLOv5-seg sperm_dataset')
    mlflow.log_params(params)
    # Train the model (MLflow logging is handled automatically)
    results = model.train(**params, project='YOLOv5-seg sperm_dataset', name='YOLOv5-seg sperm_dataset')

    # Optional: Evaluate model performance on the validation set
    val_results = model.val()
