import os
import pandas as pd

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads a csv as a pd.Dataframe
    """
    if csv_path.endswith(".xlsx"):
        pd_dataframe = pd.read_excel(csv_path)
    else:        
        pd_dataframe = pd.read_csv(csv_path)
    return pd_dataframe

def get_age_from_csv(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Gets age from csv.
    """
    return dataframe[["image id", "oocyte_age"]].copy()

def pre_process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['fecundation_date'] = pd.to_datetime(
        dataframe['fecundation_date'].str.strip(),
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )
    dataframe['patient_birthday'] = pd.to_datetime(
        dataframe['patient_birthday'],
        format='%Y-%m-%d',
        errors='coerce'
    )
    dataframe = dataframe.dropna(subset=['fecundation_date', 'patient_birthday'])
    return dataframe

def build_patient_data(data):
    patient_data = {}
    for _, row in data.iterrows():
        base_image = row['image']  
        if pd.isna(base_image):  
            continue
        fecundation_date = row['fecundation_date']
        patient_birthday = row['patient_birthday']
        for i in range(10):  
            prefixed_image = f"cropped_{i}_{base_image}"
            patient_data[prefixed_image] = {
                "fecundation_date": fecundation_date,
                "patient_birthday": patient_birthday,
            }
    return patient_data

def save_csv(output_path, data_to_save):
    output_csv = os.path.join(output_path, "segmentation_features.csv")
    data_to_save.to_csv(output_csv, index=False)
    print(f"Extracción completada. Archivo guardado en: {output_csv}")
    segmentation_features = output_csv

    return segmentation_features
