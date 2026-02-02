from config_utils import load_config
from detection.model_utils import load_yolov8_detection_model
from detection.image_utils import process_images_in_folder
from segmentation.model_utils import load_yolo_segmentation_model
from segmentation.csv_utils import load_csv, pre_process_dataframe, build_patient_data, save_csv, get_age_from_csv
from segmentation.segmentation_utils import process_directory
from tab_transformer.TabTransformer import load_pipeline_and_infer
from tab_transformer.TabTransformer_utils import feature_dt_prepreprocessing, get_transformer_model, train_model, get_top_features
import torch


def main():
    """
    Main function to process images based on configuration.
    """
    config = load_config("config.json")

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    detection_model = load_yolov8_detection_model(config["detection_model_path"], device)
    if detection_model is None:
        print("Failed to load the model. Exiting.")
        return
    input_images_to_crop_path = config["input_images_to_crop_path"]
    output_detection_dir = config["output_detection_dir"]
    
    conf_thresh = config["conf_thresh"]
    cropped_images = process_images_in_folder(input_images_to_crop_path, detection_model, output_detection_dir, device, conf_thresh)
    input_segmentation_dir = output_detection_dir + "cropped_embryos"

    segmentation_model = load_yolo_segmentation_model(config["segmentation_model_path"], device)
    if config["load_csv_with_age"]:
        data = load_csv(config["csv_with_age_path"])
        age = get_age_from_csv(data)
        patient_data = None
    else:
        data = load_csv(config["csv_path"])
        pre_processed_dataframe = pre_process_dataframe(data)
        patient_data = build_patient_data(pre_processed_dataframe)
        age = None
    output_segmentation_dir = config["output_segmentation_dir"]
    segmentation_results = process_directory(segmentation_model, input_segmentation_dir, config["segmentation_label"], patient_data, output_segmentation_dir, age)
    save_csv(config["output_csv_path"], segmentation_results)
    training_mode = config["training_mode"]
    if training_mode: 
        X, y = feature_dt_prepreprocessing(segmentation_results, data)
        X_selected = get_top_features(X,y, segmentation_results)
        tab_transformer_model_to_train = get_transformer_model()
        train_model(X_selected, X, y, tab_transformer_model_to_train)
    else:
        tab_transformer_model = config["tab_transformer_model_path"]
        scaler_path = config["scaler_path"]
        results = load_pipeline_and_infer(tab_transformer_model, segmentation_results, scaler_path, num_continuous=15)
        results.to_csv("results_timelapse_image_metadata_30m_118-128-fixed.csv", index=False)
    


if __name__ == "__main__":
    main()
