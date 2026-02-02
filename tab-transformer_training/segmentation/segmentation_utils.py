import os
import pandas as pd
from math import sqrt
import numpy as np
import cv2
from skimage.measure import regionprops, label

# Cálculo de distancia entre dos puntos
def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Función para calcular características de cada zona segmentada
def calculate_features(mask, label_name):
    mask_binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            f"{label_name}_area": 0,
            f"{label_name}_perimeter": 0,
            f"{label_name}_circularity": 0,
            f"{label_name}_eccentricity": 0,
            f"{label_name}_major_radius": 0,
            f"{label_name}_minor_radius": 0,
            f"{label_name}_centroid_x": 0,
            f"{label_name}_centroid_y": 0,
        }

    properties = regionprops(label(mask_binary))[0]
    area = properties.area
    perimeter = properties.perimeter
    centroid = properties.centroid
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    eccentricity = properties.eccentricity

    distances = [
        abs(cv2.pointPolygonTest(contours[0], (int(centroid[1]), int(centroid[0])), True))
        for point in contours[0]
    ]
    major_radius = max(distances) if distances else 0
    minor_radius = min(distances) if distances else 0

    return {
        f"{label_name}_area": area,
        f"{label_name}_perimeter": perimeter,
        f"{label_name}_circularity": circularity,
        f"{label_name}_eccentricity": eccentricity,
        f"{label_name}_major_radius": major_radius,
        f"{label_name}_minor_radius": minor_radius,
        f"{label_name}_centroid_x": centroid[0],
        f"{label_name}_centroid_y": centroid[1],
    }


def save_segmented_image(image_path, masks, classes, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio si no existe
    original_image = cv2.imread(image_path)

    # Crear una copia de la imagen para superponer las máscaras
    segmented_image = np.zeros_like(original_image)

    # Colores para cada clase
    colors = [
        (255, 0, 0),  # Clase 0: Rojo - Zona Pellucida (ZP)
        (0, 0, 255),  # Clase 1: Azul - Trophectoderm (TE)
        (0, 255, 0),  # Clase 2: Verde - Blastocoel (BC)
        (255, 255, 0)  # Clase 3: Cian - Inner Cell Mass (ICM)
    ]

    for i, mask in enumerate(masks):
        class_id = int(classes[i])
        color = colors[class_id % len(colors)]
        mask_binary = mask.cpu().numpy().astype(np.uint8)

        # Redimensionar la máscara para que coincida con la imagen original
        mask_resized = cv2.resize(mask_binary, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Aplicar color donde la máscara es activa
        segmented_image[mask_resized > 0] = color

    # Guardar la imagen segmentada
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, segmented_image)
    print(f"Imagen segmentada guardada en: {output_path}")

def process_directory(model, directory, label, patient_data, output_dir="segmented_images", age=None):
    results = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        filename_no_extension = os.path.splitext(file_name)[0]
        if not os.path.isfile(file_path):
            continue
        if not file_name in age["image id"].values:
            continue

        print(f"Procesando {file_name}...")

        # Realizar predicción con YOLO
        yolo_result = model.predict(file_path, verbose=False)

        # Verificar si se encontraron máscaras
        if yolo_result[0].masks is None:
            print(f"Sin máscaras detectadas para {file_name}. Registro omitido.")
            continue
        masks = yolo_result[0].masks.data  
        classes = yolo_result[0].boxes.cls.cpu().numpy()  
        # Crear y guardar la imagen segmentada
        save_segmented_image(file_path, masks, classes, output_dir)

        # Crear un diccionario para almacenar características
        features = {"image": file_name, "label": label}

        # Calcular edad del paciente
        if age is not None:
            patient_age = age.loc[age["image id"] == file_name, "oocyte_age"].values[0]
            features["patient_age"] = patient_age
        else:
            patient_info = patient_data.get(file_name, {})
            if patient_info:
                try:
                    features["patient_age"] = (patient_info["fecundation_date"] - patient_info["patient_birthday"]).days / 365.25
                except Exception as e:
                    print(f"Error calculando la edad para {file_name}: {e}")
                    features["patient_age"] = None
        # Almacenar centroides y áreas para relaciones entre zonas
        centroids = {}
        areas = {}

        # Calcular características para cada clase segmentada
        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            class_name = model.names[class_id]
            mask_binary = mask.cpu().numpy().astype(np.uint8)
            zone_features = calculate_features(mask_binary, class_name)
            features.update(zone_features)
            centroids[class_name] = (
                zone_features[f"{class_name}_centroid_x"], zone_features[f"{class_name}_centroid_y"])
            areas[class_name] = zone_features[f"{class_name}_area"]

        # Relación de áreas y distancias entre zonas
        if "TE" in areas and "ICM" in areas:
            features["TE_ICM_area_ratio"] = areas["TE"] / areas["ICM"] if areas["ICM"] > 0 else 0
        if "BC" in areas and "ZP" in areas:
            features["BC_ZP_area_ratio"] = areas["BC"] / areas["ZP"] if areas["ZP"] > 0 else 0
        if "TE" in centroids and "ICM" in centroids:
            features["TE_ICM_distance"] = calculate_distance(centroids["TE"], centroids["ICM"])
        results.append(features)
        results_dataframe = pd.DataFrame(results)
        results_dataframe = results_dataframe.fillna(0)
        # Add new statistical features
        required_columns = ['ICM_area', 'TE_ICM_area_ratio', 'BC_area']
        valid_columns = [col for col in required_columns if col in results_dataframe.columns]
        if valid_columns:
            #print(valid_columns)
            results_dataframe['mean_area'] = results_dataframe[valid_columns].mean(axis=1)
            #print(results_dataframe['mean_area'])
            results_dataframe['std_area'] = results_dataframe[valid_columns].std(axis=1)
            #print(results_dataframe['std_area'])
            results_dataframe['area_range'] = (results_dataframe[valid_columns].max(axis=1) -
                                            results_dataframe[valid_columns].min(axis=1))
            #print( results_dataframe['area_range'])
        else:
            results_dataframe['mean_area'] = np.nan
            results_dataframe['std_area'] = np.nan
            results_dataframe['area_range'] = np.nan

    results_dataframe.to_csv("image_metadata_30m_118-128-fixed_segmentation_results_19-10-2025.csv")
        
    return results_dataframe