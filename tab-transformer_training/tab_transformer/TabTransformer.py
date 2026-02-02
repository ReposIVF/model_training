from typing import Tuple, Union
import torch
from tab_transformer_pytorch import TabTransformer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class TabTransformerInference:
    """
    Module to handle loading and inference for a TabTransformer model.
    """

    def __init__(self, model_path: str, num_continuous: int, scaler_path, device: Union[str, torch.device] = None):
        """
        Initializes the inference module.

        Args:
            model_path (str): Path to the saved model.
            num_continuous (int): Number of continuous features in the input.
            device (Union[str, torch.device], optional): Device to load the model on. Defaults to 'cuda' if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = self.load_scaler(scaler_path)
        self.model = TabTransformer(
            categories=tuple(),
            num_continuous=num_continuous,
            dim=32,
            dim_out=1,
            depth=4,
            heads=2,
            attn_dropout=0.2,
            ff_dropout=0.2,
            mlp_hidden_mults=(4, 2),
            mlp_act=torch.nn.ReLU(),
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def load_scaler(self, scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler

    
    def preprocess(self, data: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
        """
        Preprocesses the input data.

        Args:
            data (pd.DataFrame): Input data as a pandas DataFrame.
            scaler (StandardScaler): Pre-trained scaler to normalize the data.

        Returns:
            torch.Tensor: Preprocessed data as a tensor.
        """
        top_features = ['patient_age', 'ZP_area', 'area_range', 'std_area', 'BC_area',
       'mean_area', 'ZP_perimeter', 'TE_ICM_distance', 'TE_centroid_y',
       'BC_perimeter', 'BC_ZP_area_ratio', 'ICM_perimeter', 'ICM_area',
       'ICM_circularity', 'ICM_eccentricity']
        data = data[top_features].copy()
        numeric_columns = data.select_dtypes(include=["number"]).columns 
        data.loc[:, numeric_columns] = data[numeric_columns].fillna(0)
        data_scaled = scaler.transform(data)
        return torch.tensor(data_scaled, dtype=torch.float32, device=self.device)

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Perform inference using the TabTransformer model.

        Args:
            input_data (torch.Tensor): Preprocessed input data.

        Returns:
            torch.Tensor: Predicted output.
        """
        with torch.no_grad():
            predictions = self.model(torch.empty(input_data.size(0), 0, device=self.device), input_data)
        return predictions.squeeze()

# Example usage
def load_pipeline_and_infer(
    model_path: str, 
    input_data: pd.DataFrame, 
    scaler_path: str,
    num_continuous: int
) -> pd.DataFrame:
    """
    Loads the model and performs inference on the input data.

    Args:
        model_path (str): Path to the saved TabTransformer model.
        input_data (pd.DataFrame): Input data for prediction.
        scaler (StandardScaler): Pre-trained scaler for normalizing input data.
        num_continuous (int): Number of continuous features in the input.

    Returns:
        pd.DataFrame: DataFrame containing predictions.
    """
    inference_module = TabTransformerInference(model_path, num_continuous, scaler_path)
    processed_data = inference_module.preprocess(input_data, inference_module.scaler)
    predictions = inference_module.predict(processed_data)
    input_data["predictions"] = predictions.cpu().numpy()
    results = input_data[["image", "predictions"]].copy()
    #return pd.DataFrame({"Prediction": predictions.cpu().numpy()})
    return results
