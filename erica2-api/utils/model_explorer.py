import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

load_dotenv()

# Default local fallback paths
DEFAULT_PATHS = {
    "cropper": "./models/erica_cropper.pt",
    "segmentation": "./models/erica_segmentor_n.pt",
    "scoring": "./models/erica_model2.pth",
    "scaler": "./models/scaler_info.json"
}

# S3 Config
S3_BUCKET = "erica-models-ivf"
AWS_ACCESS_KEY = os.getenv("ERICA_AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("ERICA_AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("ERICA_BUCKET_REGION")


def get_model_file(key: str, filename: str = None) -> str:
    """
    Generic function to retrieve model/scaler file. If not found locally, attempts to download from S3.

    Args:
        key (str): Type of model, used to get default if not provided (e.g., "cropper", "scoring").
        filename (str): S3 object name (optional). If None, uses default.

    Returns:
        str: Local path to the model or scaler.
    """
    if not filename:
        print(f"[INFO] No filename provided. Using default for {key}")
        return DEFAULT_PATHS[key]

    local_path = os.path.join("./models", filename)

    if verify_file(local_path):
        print(f"[INFO] {filename} already exists locally.")
        return local_path

    try:
        download_file_s3(filename, local_path)
        return local_path
    except Exception as e:
        print(f"[WARNING] Failed to download {filename}. Using default for {key}: {e}")
        return DEFAULT_PATHS[key]


def download_file_s3(s3_key: str, local_path: str):
    """
    Downloads a file from S3 to the given local path.

    Args:
        s3_key (str): Key of the file in S3.
        local_path (str): Path to store the file locally.
    """
    print(f"[INFO] Downloading {s3_key} from S3 → {local_path}")

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    try:
        s3.download_file(S3_BUCKET, s3_key, local_path)
        print(f"[INFO] Download complete: {local_path}")
    except FileNotFoundError:
        raise RuntimeError("Local path not found.")
    except (NoCredentialsError, PartialCredentialsError):
        raise RuntimeError("AWS credentials error.")
    except Exception as e:
        raise RuntimeError(f"Unexpected S3 error: {e}")


def verify_file(filepath: str) -> bool:
    """Check if a local file exists."""
    return os.path.exists(filepath)


# Specific accessors
def get_cropper_model_weights(models: list) -> str:
    """Get cropper model weights from models list."""
    filename = next((item['filename'] for item in models if item['type'] == 'cropper'), None)
    if not filename:
        print(f"[INFO] No filename found for cropper model. Using default.")
        return DEFAULT_PATHS["cropper"]
    print(f"[INFO] Cropper model filename: {filename}")
    return get_model_file("cropper", filename)

def get_segmentation_model_weights(models: list) -> str:
    """Get segmentation model weights from models list."""
    filename = next((item['filename'] for item in models if item['type'] == 'segmentation'), None)
    print(f"[INFO] Segmentation model filename: {filename}")
    if not filename:
        print(f"[INFO] No filename found for segmentation model. Using default.")
        return DEFAULT_PATHS["segmentation"]
    print(f"[INFO] Segmentation model filename: {filename}")
    return get_model_file("segmentation", filename)

def get_score_model_weights(models: list) -> str:
    """Get scoring model weights from models list."""
    filename = next((item['filename'] for item in models if item['type'] == 'scoring'), None)
    print(f"[INFO] Scoring model filename: {filename}")
    if not filename:
        print(f"[INFO] No filename found for scoring model. Using default.")
        return DEFAULT_PATHS["scoring"]
    print(f"[INFO] Scoring model filename: {filename}")
    return get_model_file("scoring", filename)

def get_scaler_info(models: list) -> str:
    """Get scaler info from models list."""
    print(f"[INFO] Models: {models}")
    filename = next((item['filename'] for item in models if item['type'] == 'scaler'), None)
    print(f"[INFO] Scaler model filename: {filename}")
    if not filename:
        print(f"[INFO] No filename found for scaler model. Using default.")
        return DEFAULT_PATHS["scaler"]
    print(f"[INFO] Scaler model filename: {filename}")
    return get_model_file("scaler", filename)