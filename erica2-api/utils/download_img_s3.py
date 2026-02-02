import os
import sys
import shutil
import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_s3_config

def download_images_from_s3(embryos: list, cycle_id: str) -> list:
    """
    Download embryo images from S3 and update local paths. The output folder is cleaned/recreated each time.

    Args:
        embryos (list): List of embryo dicts with 'image' field (S3 key).
        cycle_id (str): ID of the cycle (used as local folder name).

    Returns:
        list: Updated embryo dicts with local image paths.
    """
    s3_config = get_s3_config()
    bucket_name = s3_config['bucket']
    aws_key = s3_config['aws_access_key_id']
    aws_secret = s3_config['aws_secret_access_key']
    region = s3_config['region']
    
    print(f"[download_img_s3] S3 Config:")
    print(f"[download_img_s3]   Bucket: {bucket_name}")
    print(f"[download_img_s3]   Region: {region}")
    print(f"[download_img_s3]   Access Key: {aws_key[:4]}...{aws_key[-4:] if aws_key else 'NOT SET'}")
    print(f"[download_img_s3]   Secret Key: {'***' if aws_secret else 'NOT SET'}")

    # Initialize S3 client
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=region
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise RuntimeError(f"[ERROR] AWS credentials error: {str(e)}")

    folder_path = f'./{cycle_id}'

    # Replace folder if it exists
    if os.path.exists(folder_path):
        print(f"[INFO] Removing existing folder: {folder_path}")
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    downloaded_embryos = []

    print(f"[INFO] Downloading images for cycle {cycle_id} from S3...")

    for embryo in embryos:
        try:
            image_key = embryo.get('image')
            local_path = os.path.join(folder_path, os.path.basename(image_key))

            print(f"[INFO] → Downloading: {image_key} → {local_path}")
            s3.download_file(bucket_name, image_key, local_path)

            downloaded_embryos.append({
                "embryo": embryo.get('embryo'),
                "image": local_path,
                "isEmbryo": embryo.get('isEmbryo', False),
                "pgt": embryo.get('pgt', "")
            })

        except ClientError as e:
            print(f"[WARNING] Failed to download {image_key}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error with {image_key}: {e}")
            continue

    print(f"[INFO] Total images downloaded: {len(downloaded_embryos)}")
    return downloaded_embryos