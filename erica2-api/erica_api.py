"""
@author: Alberto Leon
@description: Lambda function handler for the Erica ranking/evaluation service
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_validation_key
from utils.get_embryos_db import get_embryos_from_database, get_clinic_models
from utils.download_img_s3 import download_images_from_s3
from utils.erica_pipeline import erica_pipeline
from utils.set_rank_db import set_ranking_on_database
from utils.clean_files import clear_folder


def erica_api(event):
    """
    Lambda function handler for the Erica ranking/evaluation service.

    Args:
        event (dict): Contains 'objectId' and 'validation_key'.

    Returns:
        dict: Result status and message or error.
    """
    cycle_object_id = event.get('objectId')
    validation_key = event.get('validation_key')

    if not cycle_object_id or not validation_key:
        return {
            "status": 400,
            "error": "Missing 'objectId' or 'validation_key'."
        }

    expected_key = get_validation_key()
    if validation_key != expected_key:
        return {
            "status": 403,
            "error": "Invalid validation key."
        }

    try:
        print(f"[INFO] Starting evaluation for cycle {cycle_object_id}")

        # Step 1: Cleanup temp folder
        clear_folder(cycle_object_id)

        # Step 2: Retrieve embryos and clinic data
        embryos, mother_age, oocyte_origin, clinic_id = get_embryos_from_database(cycle_object_id)
        print(f"[INFO] Retrieved {len(embryos)} embryos from DB")

        # Step 3: Get model paths
        models = get_clinic_models(clinic_id)
        print(f"[INFO] Models: {models}")

        # Step 4: Download embryo images
        embryos = download_images_from_s3(embryos, cycle_object_id)
        print(f"[INFO] Downloaded {len(embryos)} images")

        # Step 5: Process pipeline
        ranked_embryos = erica_pipeline(embryos, mother_age, oocyte_origin, models)
        print("[INFO] Ranking complete.")

        # Step 6: Store results in DB
        set_ranking_on_database(ranked_embryos, cycle_object_id, models)

        # Step 7: Clean up again
        clear_folder(cycle_object_id)

        return {
            "status": 200,
            "message": "Ranking completed successfully."
        }

    except Exception as e:
        print(f"[ERROR] Error during process: {e}")
        return {
            "status": 500,
            "error": str(e)
        }