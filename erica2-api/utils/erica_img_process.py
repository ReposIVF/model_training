import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from math import sqrt
from PIL import Image
from skimage.measure import regionprops, label
from ultralytics import YOLO
from tab_transformer_pytorch import TabTransformer

def process_imagesv2(
    embryos_list,
    mother_age,
    scaler_json_path='./models/scaler_info.json',
    model_path='./models/erica_segmentor_n.pt',
    label=label
):
    """
    Process embryo images: segment zones, extract morphological features, standardize them.

    Returns:
        pd.DataFrame: standardized features with embryo metadata
    """
    embryo_results = []

    # Load segmentation model once
    print("[INFO] Loading segmentation model...")
    model = load_model("segment", model_path)

    for embryo in embryos_list:
        if not embryo.get("isEmbryo", False):
            continue

        image_path = embryo.get("image")
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            print(f"[WARNING] Unable to open image {image_path}: {e}")
            continue

        print(f"[INFO] Processing embryo image: {image_path}")
        yolo_result = model.predict(image_path, verbose=False)
        if not yolo_result or yolo_result[0].masks is None:
            print(f"[WARNING] No masks detected for {image_path}. Skipping.")
            continue

        masks = yolo_result[0].masks.data
        classes = yolo_result[0].boxes.cls.cpu().numpy()
        feature_row = create_base_feature_row(embryo, mother_age, label)

        centroids, areas = {}, {}

        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            class_name = model.names[class_id]
            mask_binary = mask.cpu().numpy().astype(np.uint8)
            zone_features = calculate_features(mask_binary, class_name)

            feature_row.update(zone_features)
            centroids[class_name] = (
                zone_features[f"{class_name}_centroid_x"],
                zone_features[f"{class_name}_centroid_y"]
            )
            areas[class_name] = zone_features[f"{class_name}_area"]

        # Relaciones derivadas
        if "TE" in areas and "ICM" in areas and areas["ICM"] > 0:
            feature_row["TE_ICM_area_ratio"] = areas["TE"] / areas["ICM"]
        else:
            feature_row["TE_ICM_area_ratio"] = 0

        if "BC" in areas and "ZP" in areas and areas["ZP"] > 0:
            feature_row["BC_ZP_area_ratio"] = areas["BC"] / areas["ZP"]
        else:
            feature_row["BC_ZP_area_ratio"] = 0

        if "TE" in centroids and "ICM" in centroids:
            feature_row["TE_ICM_distance"] = calculate_distance(centroids["TE"], centroids["ICM"])
        else:
            feature_row["TE_ICM_distance"] = 0

        embryo_results.append(feature_row)

    if not embryo_results:
        raise ValueError("No valid embryos processed.")

    df = pd.DataFrame(embryo_results)
    df = fill_missing_features(df)
    df = standardize_features(df, scaler_json_path)
    return df


def create_base_feature_row(embryo, mother_age, label):
    return {
        "embryo": embryo["embryo"],
        "isEmbryo": embryo["isEmbryo"],
        "image": embryo["image"],
        "label": label,
        "pgt": embryo.get("pgt", ""),
        "patient_age": mother_age
    }


def calculate_features(mask, label_name):
    props = {
        f"{label_name}_area": 0,
        f"{label_name}_perimeter": 0,
        f"{label_name}_circularity": 0,
        f"{label_name}_eccentricity": 0,
        f"{label_name}_major_radius": 0,
        f"{label_name}_minor_radius": 0,
        f"{label_name}_centroid_x": 0,
        f"{label_name}_centroid_y": 0,
    }

    mask_binary = mask.astype(np.uint8)
    if not np.any(mask_binary):
        return props

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = regionprops(label(mask_binary))

    if not regions:
        return props

    region = regions[0]
    area = region.area
    perimeter = region.perimeter
    centroid = region.centroid
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    distances = [
        abs(cv2.pointPolygonTest(contours[0], (int(centroid[1]), int(centroid[0])), True))
        for _ in contours[0]
    ]

    props.update({
        f"{label_name}_area": area,
        f"{label_name}_perimeter": perimeter,
        f"{label_name}_circularity": circularity,
        f"{label_name}_eccentricity": region.eccentricity,
        f"{label_name}_major_radius": max(distances) if distances else 0,
        f"{label_name}_minor_radius": min(distances) if distances else 0,
        f"{label_name}_centroid_x": centroid[0],
        f"{label_name}_centroid_y": centroid[1],
    })
    return props


def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def fill_missing_features(df):
    required_columns = [
        "ICM_area", "TE_ICM_area_ratio", "BC_area", "ZP_area", "TE_ICM_distance",
        "ICM_perimeter", "ICM_circularity", "ICM_eccentricity"
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    df['mean_area'] = df[["ICM_area", "TE_ICM_area_ratio", "BC_area"]].mean(axis=1)
    df['std_area'] = df[["ICM_area", "TE_ICM_area_ratio", "BC_area"]].std(axis=1)
    df['area_range'] = df[["ICM_area", "TE_ICM_area_ratio", "BC_area"]].max(axis=1) - \
                       df[["ICM_area", "TE_ICM_area_ratio", "BC_area"]].min(axis=1)
    return df


def standardize_features(df, scaler_json_path):
    with open(scaler_json_path, "r") as f:
        scaler = json.load(f)

    feature_names = scaler["feature_names"]
    means = np.array(scaler["mean"])
    stds = np.array(scaler["std"])

    for feature, mean, std in zip(feature_names, means, stds):
        if feature in df.columns:
            df[feature] = (df[feature] - mean) / std
    return df


def load_model(process, model_path):
    device = "cpu"
    if process == "predict":
        model = TabTransformer(
            categories=tuple(),
            num_continuous=15,
            dim=32,
            dim_out=1,
            depth=4,
            heads=2,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=torch.nn.ReLU()
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)
    elif process == "segment":
        return YOLO(model_path).to(device)
    else:
        raise ValueError("Unknown model process type")