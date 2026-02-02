import json
import os
from typing import Dict


def load_config(config_path: str) -> Dict[str, str]:
    """
    Load and validate the configuration file.

    Args:
        config_path (str): Path to the config.json file.

    Returns:
        Dict[str, str]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If required keys are missing in the configuration.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        config = json.load(file)

    required_keys = ["detection_model_path",
                     "segmentation_model_path",
                     "csv_path",
                     "input_images_to_crop_path", 
                     "output_detection_dir", 
                     "device", 
                     "conf_thresh", 
                     "segmentation_label",
                     "output_segmentation_dir",
                     "output_csv_path",
                     "tab_transformer_model_path",
                     "csv_with_age_path",
                     "load_csv_with_age",
                     "training_mode"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config
