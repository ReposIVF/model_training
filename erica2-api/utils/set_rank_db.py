"""
@author: Alberto Leon, Emanuel Martin
@description: Set embryo ranking in the database.
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, get_parse_headers

HEADERS = get_parse_headers()


def set_ranking_on_database(objects: List[dict], cycle_id: str, models: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Upload embryo ranking to the database for a given cycle.

    Args:
        objects (list): Ranked embryo objects with relevant info.
        cycle_id (str): ObjectId of the cycle.
        models (list): List of model dicts with 'id', 'type', and 'filename'.

    Returns:
        dict: Response from the database.
    """
    # Build model pointers for Parse
    model_pointers = {}
    for model in models:
        model_type = model.get('type')
        model_id = model.get('id')
        if model_type and model_id:
            # Create Parse pointer
            model_pointers[model_type] = {
                "__type": "Pointer",
                "className": "ERICAModels",
                "objectId": model_id
            }
    
    print(f"[DEBUG] Model pointers: {model_pointers}")
    
    url = f"{config.parse_server_url}/functions/setCycleEmbryos"
    payload = {
        "objects_ranking": objects,
        "objectId": cycle_id,
        "ericaVersion": "I+P",
        "apiVersion": config.version,
        "models": model_pointers
    }

    try:
        print(f"[INFO] Uploading ranking to database for cycle [{cycle_id}]...")
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"[INFO] Upload successful. Response: {data}")
        return data

    except requests.RequestException as e:
        raise RuntimeError(f"[ERROR] Failed to upload ranking to database: {e}")
