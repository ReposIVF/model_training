"""
@author: Pollo, Alberto

"""
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy.stats import norm
from ultralytics import YOLO
from tab_transformer_pytorch import TabTransformer  # Make sure this is installed
from utils.model_explorer import get_score_model_weights

def erica(embryos_list, models: list):
    """
    ERICA scoring model.
    
    Args:
        embryos_list: DataFrame with embryo features
        models (list): List of model dicts with 'id', 'type', and 'filename'
    
    Returns:
        list: Ranked embryos with scores
    """
    print('EmbryosKeys', embryos_list.columns.tolist())
    # BRAND NEW
    feature_columns = ['patient_age', 'ZP_area', 'area_range', 'std_area', 'BC_area',
                       'mean_area', 'ZP_perimeter', 'TE_ICM_distance', 'TE_centroid_y',
                       'BC_perimeter', 'BC_ZP_area_ratio', 'ICM_perimeter', 'ICM_area',
                       'ICM_circularity', 'ICM_eccentricity']
    
    # Fill missing features
    embryos_list = fill_missing_features(embryos_list)
    
    # Ensure all required columns exist
    embryos_list = ensure_required_columns(embryos_list, feature_columns)

    model_path = get_score_model_weights(models)

    # Brand new
    print("predict and rank")
    sorted_index, scores = predict_and_rank(embryos_list, model_path, feature_columns)
    
    print ("ERICA Function")
    print(scores)
    
    print("fix embryos scores 2")
    fixed_embryos_list = fix_embryos_scores_2_data(embryos_list, scores, sorted_index)
    boosted_embryos_list = boost_ploidies(fixed_embryos_list)

    return boosted_embryos_list

def fill_missing_features(df, method='zero', fill_value=0):
    if method == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif method == 'value':
        return df.fillna(fill_value)
    elif method == 'zero':
        return df.fillna(0)
    else:
        raise ValueError("Método no soportado. Usa 'mean', 'value' o 'zero'.")


def ensure_required_columns(df, required_columns, default_value=0):
    """
    Ensure all required columns exist in the DataFrame.
    Add missing columns with default values.
    
    Args:
        df: DataFrame to check
        required_columns: List of column names that must exist
        default_value: Value to fill missing columns (default: 0)
    
    Returns:
        DataFrame with all required columns
    """
    for col in required_columns:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in data. Filling with {default_value}")
            df[col] = default_value
    
    return df


def fix_embryos_scores_2_data(embryos_list, scores, sorted_scores):
    """
        description: fix the scores to the embryos list.
    
    """
    embryos_to_list = embryos_list.to_json(orient="records")
    json_data = json.loads(embryos_to_list)

    for i, embryo in enumerate(json_data):
        if isinstance(scores, np.ndarray):
            embryo["score"] = int(scores[i])
        else:
            embryo["score"] = int(scores)

    return json_data

def boost_ploidies(embryos_list):
    """
        description: boost the ploidies of the embryos.
        If 'pgt' field is missing or has unexpected values, embryo is not boosted.
    """
    alpha = 0.3  # Boosting factor for euploid
    beta = 0.7   # Penalizing factor for aneuploid

    for embryo in embryos_list:
        try:
            pgt = embryo.get('pgt', None)
            if pgt == 'euploid':
                embryo['score'] = int((1 - alpha) * embryo['score'] + alpha * 100)
            elif pgt == 'aneuploid':
                embryo['score'] = int((1 - beta) * embryo['score'])
            # If pgt is None or other value, score remains unchanged
        except Exception as e:
            print(f"[WARNING] Error boosting ploidy for embryo {embryo.get('id', 'unknown')}: {str(e)}")
            # Continue without boosting this embryo

    return embryos_list


def predict_and_rank(dataframe, model_path, feature_columns):
    """
    Predict using a TabTransformer model, calculate scores, and return sorted indices.
    If prediction fails, returns all zeros for scores.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing features.
        model_path (str): Path to the pre-trained TabTransformer model file.
        feature_columns (list): List of column names to be used as features.

    Returns:
        np.ndarray: Indices of the rows sorted from the highest to the lowest score.
        np.ndarray: Scores corresponding to the predictions.
    """
    try:
        # Load the TabTransformer model using the provided load_model function
        model = load_model("predict", model_path)

        # Determine the device
        device = next(model.parameters()).device

        # Extract features and convert to tensor
        try:
            features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32).to(device)
        except KeyError as e:
            print(f"[ERROR] Missing columns in data: {e}")
            print(f"[WARNING] Available columns: {dataframe.columns.tolist()}")
            print(f"[WARNING] Required columns: {feature_columns}")
            # Return default scores (0) for all embryos
            num_embryos = len(dataframe)
            default_scores = np.zeros(num_embryos)
            sorted_indices = np.arange(num_embryos)
            return sorted_indices, default_scores

        # Perform predictions
        with torch.no_grad():
            predictions = model(
                torch.empty(features.size(0), 0).to(features.device),
                features
            ).squeeze().cpu().numpy()

        if predictions.ndim == 0:
            predictions = np.array([predictions])

        # Look if any prediction is negative or greater than 1
        if np.any(predictions < 0) or np.any(predictions > 1):
            with torch.no_grad():
                raw_outputs = model(
                    torch.empty(features.size(0), 0).to(features.device),
                    features
                )
                probs = torch.sigmoid(raw_outputs)
                predictions = probs.squeeze().cpu().numpy()

            if predictions.ndim == 0:
                predictions = np.array([predictions])
                
        # Calculate scores using sigmoid transformation
        scores = set_predictions_0_100(predictions)
        # scores = sigmoid_score(predictions)
        # Sort indices by score from highest to lowest
        sorted_indices = np.argsort(-scores)

        return sorted_indices, scores
    
    except Exception as e:
        print(f"[ERROR] Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default scores (0) for all embryos on error
        num_embryos = len(dataframe)
        default_scores = np.zeros(num_embryos)
        sorted_indices = np.arange(num_embryos)
        return sorted_indices, default_scores


def set_predictions_0_100(predictions):
    """
    This function transforms the predictions to a 0-100 scale, from 0 to 1.
    """

    print("Set predictions")
    print("Original type:", type(predictions))
    print("Original value:", predictions)

    # Convert predictions to a NumPy array if it's not already one
    if isinstance(predictions, (int, float)):  # If it's a single number, make it an array
        predictions = np.array([predictions])
    elif not isinstance(predictions, np.ndarray):  # If it's not a NumPy array, convert it
        predictions = np.array(predictions)

    print("Converted type:", type(predictions))
    print("Converted value:", predictions)

    # Perform the transformation
    predictions = predictions * 100

    return predictions



def sigmoid_score(predictions, k=5):
    """
    Apply a sigmoid transformation to scale predictions to scores.
    """
    scores = 100 / (1 + np.exp(-k * (predictions - 0.5)))
    return scores



def load_model(process, model_path):
    """
    Load a model (either TabTransformer or YOLO v8) with pre-trained weights.

    Args:
        model_type (str): The type of model to load ('tabtransformer' or 'yolo').
        model_path (str): Path to the model file.

    Returns:
        torch.nn.Module or YOLO: Loaded model instance.
    """
    device = "cpu"
    # print(f"Using device: {device}")

    if process.lower() == "predict":
        # Initialize the TabTransformer model (ensure you define it beforehand)
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
        # Load pre-trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model.to(device)
    
    elif process.lower() == "segment":
        # Load YOLO v8 model
        return YOLO(model_path).to(device)
    
    else:
        raise ValueError("Invalid process. Use 'predict' or 'segment'.")
