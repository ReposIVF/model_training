"""
@author: Alberto Leon, Emanuel Martin
@description: Get cycle and model data from the database.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, get_parse_headers

BASE_URL = f"{config.parse_server_url}/functions"
HEADERS = get_parse_headers()

# Log Parse configuration on module load
print(f"[get_embryos_db] ========== PARSE CONFIGURATION ==========")
print(f"[get_embryos_db] Base URL: {BASE_URL}")
print(f"[get_embryos_db] Headers:")
for key, value in HEADERS.items():
    if 'Key' in key or 'key' in key:
        masked = value[:4] + '...' + value[-4:] if len(value) > 8 else '***'
        print(f"[get_embryos_db]   {key}: {masked}")
    else:
        print(f"[get_embryos_db]   {key}: {value}")
print(f"[get_embryos_db] ============================================\n")


def get_clinic_models(clinic_id: str) -> List[Dict[str, str]]:
    """
    Get the clinic models from the database given a clinic_id.
    
    Args:
        clinic_id (str): ID of the clinic.
        
    Returns:
        list: List of model dicts with 'id', 'type', and 'filename'.
    """
    url = f"{BASE_URL}/getClinicModels"
    payload = {"clinicId": clinic_id}

    try:
        print(f"[INFO] Getting models for clinic [{clinic_id}]...")
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json().get('result', {})
        models = result.get('result', [])
        print(f"[DEBUG] Data retrieved: {models}")
        
        return models

    except requests.RequestException as e:
        print(f"[ERROR] Network error while getting models: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    
    return []


def get_embryos_from_database(object_id: str) -> Tuple[List[Dict], int, str, str]:
    """
    Get embryo metadata from the database given a cycle object_id.

    Args:
        object_id (str): Cycle objectId.

    Returns:
        tuple: (List of embryo dicts, mother's age, oocyte origin, clinicId)
    """
    fetch_url = f"{BASE_URL}/getCycleEmbryos"
    status_url = f"{BASE_URL}/setCycleStatus"
    payload = {"objectId": object_id}

    try:
        print(f"[INFO] Retrieving data for cycle [{object_id}]...")
        print(f"[DEBUG] Request URL: {fetch_url}")
        print(f"[DEBUG] Request payload: {payload}")
        print(f"[DEBUG] Request headers: {', '.join([f'{k}={v[:10]}...' if 'Key' in k else f'{k}={v}' for k, v in HEADERS.items()])}")
        
        response = requests.post(fetch_url, headers=HEADERS, json=payload)
        
        print(f"[DEBUG] Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] Response body: {response.text}")
        
        response.raise_for_status()
        result = response.json().get('result', {})
        data = result.get('result', {})

        print(f"[DEBUG] Data retrieved: {data}")

        embryos = data.get('embryos', [])
        clinic_id = data.get('clinicId', '')
        mother_age = data.get('mothersAge', 0)
        oocyte_origin = data.get('eggOrigin', '')

        print(f"[DEBUG] Clinic ID: {clinic_id}, Mother age: {mother_age}, Oocyte origin: {oocyte_origin}")
        print(f"[DEBUG] Embryos: {embryos}")

        embryo_list = [{
            "embryo": embryo.get('objectId'),
            "image": f"{embryo.get('uuid')}.{embryo.get('imageFormat')}",
            "isEmbryo": False,
            "pgt": embryo.get('pgt', "")
        } for embryo in embryos]

        print(f"[DEBUG] Embryo list: {embryo_list}")

        print(f"[INFO] Retrieved {len(embryo_list)} embryos.")
        print(f"[INFO] Mother age: {mother_age}, Egg origin: {oocyte_origin}, Clinic ID: {clinic_id}")

        # Set cycle status to "Processing"
        status_payload = {"cycleId": object_id, "status": "Processing"}
        status_response = requests.post(status_url, headers=HEADERS, json=status_payload)
        status_response.raise_for_status()
        print(f"[INFO] Cycle status updated: {status_response.json()}")

        return embryo_list, mother_age, oocyte_origin, clinic_id

    except requests.RequestException as e:
        raise RuntimeError(f"[ERROR] Network error while fetching embryo data: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error: {e}")
