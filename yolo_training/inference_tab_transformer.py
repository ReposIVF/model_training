import torch
import pandas as pd
from torch import nn
from tab_transformer_pytorch import TabTransformer
from typing import List
import logging
import numpy as np
import json

logging.basicConfig(level=logging.INFO)

# DATA TO STANDARDIZE
ENTRY_DATA = "data/final_standardized_output_from_latest.csv"
MOTILITY_JSON_PATH = "data/standardizer_motility_data.json" 
MORPHOLOGY_JSON_PATH = "data/standardizer_morpho_data.json"
OUTPUT_STANDARDIZE_PATH = "data/final_combined_standardized_output_full_pipeline.csv"
MODEL_PATH = 'tab_transformer.pth'
THRESHOLD = 0.5
DEVICE = "cuda"
FEATURE_COLUMNS = [
    'sta_VSL', 'sta_VCL', 'sta_HMP', 'sta_orientated_angle_mean',
    'sta_circularity_mean', 'sta_convexity_mean', 'sta_compactness_mean',
    'sta_minor_axis_radius_mean'
]
LABEL_MAPPING = {
    1: 'Observable ICM and TE',
    0: 'Otherwise'
}

with open(MOTILITY_JSON_PATH, "r") as f:
    motility_params = json.load(f)

with open(MORPHOLOGY_JSON_PATH, "r") as f:
    morphology_params = json.load(f)

target_features = [
    'VSL', 'VCL', 'HMP',
    'orientated_angle_mean', 'circularity_mean',
    'convexity_mean', 'compactness_mean', 'minor_axis_radius_mean'
]

motility_features = {'VSL', 'VCL', 'HMP'}
morphology_features = set(target_features) - motility_features

def yeo_johnson_transform(x, lmbda):
    """
    Applies the Yeo-Johnson power transformation to a NumPy array.

    This transformation is an extension of the Box-Cox transformation 
    to handle both positive and negative values.

    Args:
        x (np.ndarray): Input array of numeric values.
        lmbda (float): Lambda parameter for the transformation.

    Returns:
        np.ndarray: Transformed array with the same shape as `x`.
    """
    x = np.asarray(x)
    pos = x >= 0
    out = np.zeros_like(x, dtype=np.float64)
    if lmbda != 0:
        out[pos] = ((x[pos] + 1) ** lmbda - 1) / lmbda
    else:
        out[pos] = np.log(x[pos] + 1)
    if lmbda != 2:
        out[~pos] = -((-x[~pos] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
    else:
        out[~pos] = -np.log(-x[~pos] + 1)
    return out

def apply_full_standardization_pipeline(df):
    """
    Applies a full standardization pipeline to the input DataFrame using predefined transformation parameters.

    Each feature value in the DataFrame is transformed using a Yeo-Johnson transformation 
    followed by standardization (mean and scale), based on a configuration dictionary 
    indexed by metadata keys derived from row-specific information.

    Args:
        df (pd.DataFrame): Input DataFrame with columns required to generate metadata keys 
                           and target features to standardize.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each standardized feature, 
                      prefixed by 'sta_'.
    """
    df_std = df.copy()
    for i, row in df.iterrows():
        key_mot = f"{row['magnification'].lower()}-" \
                  f"{str(row['video_resolution']).replace('.0','').replace('[','').replace(']','').replace(', ','_')}-" \
                  f"{row['pvp']}"
        key_morph = f"('{row['magnification']}', '{row['video_resolution']}', '{row['pvp']}')"

        for feature in target_features:
            val = row.get(feature, np.nan)
            if pd.isna(val):
                continue

            if feature in motility_features:
                cfg = motility_params.get(key_mot, {}).get(feature, {})
                lmbda, mean, scale = cfg.get("lambdas_"), cfg.get("mean_"), cfg.get("scale_")
            else:
                cfg = morphology_params.get(key_morph, {}).get(feature, {})
                lmbda, mean, scale = cfg.get("lambda"), cfg.get("mean"), cfg.get("scale")

            if None in (lmbda, mean, scale):
                continue

            try:
                transformed = yeo_johnson_transform([val], lmbda)[0]
                standardized = (transformed - mean) / scale
                df_std.at[i, f"sta_{feature}"] = standardized
            except Exception:
                df_std.at[i, f"sta_{feature}"] = np.nan
    return df_std



def load_model(num_features: int) -> TabTransformer:
    """
    Load a trained TabTransformer model.

    Args:
        num_features (int): Number of continuous input features.

    Returns:
        TabTransformer: The loaded model.
    """
    model = TabTransformer(
        categories=tuple(),
        num_continuous=num_features,
        dim=32,
        dim_out=1,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU()
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.eval()
    return model


def read_inference_data(csv_path: str, features: List[str]) -> torch.Tensor:
    """
    Read CSV and extract features for inference.

    Args:
        csv_path (str): Path to CSV file.
        features (List[str]): List of feature column names.

    Returns:
        torch.Tensor: Tensor of input features.
    """
    data = pd.read_csv(csv_path)
    if not all(col in data.columns for col in features):
        missing = set(features) - set(data.columns)
        raise ValueError(f"Missing required features: {missing}")
    return torch.tensor(data[features].values, dtype=torch.float32), data


def predict(model: TabTransformer, input_tensor: torch.Tensor, threshold: float = THRESHOLD) -> List[str]:
    """
    Perform prediction using TabTransformer model.

    Args:
        model (TabTransformer): Trained model.
        input_tensor (torch.Tensor): Input tensor.
        threshold (float): Threshold to classify output.

    Returns:
        List[str]: List of predicted labels.
    """
    with torch.no_grad():
        dummy_categorical = torch.empty(input_tensor.size(0), 0)
        logits = model(dummy_categorical, input_tensor)
        probs = torch.sigmoid(logits).squeeze()
        predictions = (probs > threshold).int().tolist()
        return [LABEL_MAPPING[pred] for pred in predictions]


def run_inference_pipeline(model_path: str, data_path: str, features: List[str]) -> pd.DataFrame:
    """
    Run full inference pipeline and return dataframe with predictions.

    Args:
        model_path (str): Path to saved model.
        data_path (str): Path to inference CSV.
        features (List[str]): List of feature names.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    model = load_model(num_features=len(features))
    input_tensor, original_data = read_inference_data(data_path, features)
    predictions = predict(model, input_tensor)
    original_data['prediction'] = predictions
    return original_data


if __name__ == '__main__':
    df = pd.read_csv(ENTRY_DATA)
    df_combined = apply_full_standardization_pipeline(df)
    df_combined.to_csv(OUTPUT_STANDARDIZE_PATH, index=False)

    try:
        results_df = run_inference_pipeline(MODEL_PATH, OUTPUT_STANDARDIZE_PATH, FEATURE_COLUMNS)
        logging.info("Inference completed successfully.")
        results_df.to_csv('inference_results.csv', index=False)
    except Exception as e:
        logging.error(f"Inference failed: {e}")
