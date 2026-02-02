from utils.erica_img_process import process_imagesv2
from utils.erica_model import erica
from utils.erica_cropper import cropper
from utils.model_explorer import get_scaler_info, get_segmentation_model_weights
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

ALPHABET = [chr(i) for i in range(65, 91)]  # A-Z


def erica_pipeline(
    embryos_list: List[Dict],
    mother_age: int,
    oocyte_origin: str,
    models: List[Dict[str, str]]
) -> List[Dict]:
    """
    Pipeline for the Erica ranking/evaluation service.
    Continues processing even if intermediate steps fail, assigning default scores.

    Args:
        embryos_list (list): List of embryo metadata and image paths.
        mother_age (int): Age of the patient.
        oocyte_origin (str): Egg origin (Autologous, Donor, etc.).
        models (list): List of model dicts with 'id', 'type', and 'filename'.

    Returns:
        list: Ranked embryos with essential information.
    """
    print("[INFO] Starting Erica pipeline...")

    try:
        # Step 1: Load model metadata
        scaler_path = get_scaler_info(models)
        segmentor_path = get_segmentation_model_weights(models)

        # Step 2: Crop embryos from raw images
        print("[INFO] Cropping images...")
        embryos_list = cropper(embryos_list, models)
        if not embryos_list:
            raise ValueError("No embryos returned after cropping.")

        print("[DEBUG] Post-crop keys:", embryos_list[0].keys())

        # Step 3: Process images into feature dataframe
        print("[INFO] Extracting features from images...")
        try:
            embryos_df = process_imagesv2(embryos_list, mother_age, scaler_path, model_path=segmentor_path)
            print("[DEBUG] Feature columns:", embryos_df.columns.tolist())
        except Exception as e:
            print(f"[WARNING] Error processing images, using default scores: {str(e)}")
            # Create default results with score 0
            ranked_embryos = create_default_ranked_embryos(embryos_list)
            return complete_ranking_pipeline(ranked_embryos)

        # Step 4: Predict and rank using Erica model
        print("[INFO] Running ranking model...")
        try:
            ranked_embryos = erica(embryos_df, models)
        except Exception as e:
            print(f"[WARNING] Error during ranking, using default scores: {str(e)}")
            # Use default scores (0) for all embryos
            ranked_embryos = create_default_ranked_embryos(embryos_list)
        
        print("[DEBUG] Ranking complete")

        # Step 5: Sort by score
        print("[INFO] Sorting embryos by score...")
        sorted_embryos = sorted(ranked_embryos, key=lambda x: x['score'], reverse=True)

        for i, embryo in enumerate(sorted_embryos):
            embryo['letter'] = ALPHABET[i] if i < len(ALPHABET) else f"Z{i - 25}"

        # Step 6: Reduce to essential info
        result = keep_essential_info(sorted_embryos)
        print("[INFO] Pipeline complete. Ranked embryos:")
        for emb in result:
            print(emb)

        return result
    
    except Exception as e:
        print(f"[ERROR] Critical error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default ranked embryos
        return create_default_ranked_embryos(embryos_list)


def create_default_ranked_embryos(embryos_list: List[Dict]) -> List[Dict]:
    """
    Create a default ranked list with score 0 for all embryos.
    Used when processing fails.
    
    Args:
        embryos_list: Original list of embryos
    
    Returns:
        list: Embryos with default score 0
    """
    ranked = []
    for emb in embryos_list:
        ranked.append({
            "embryo": emb.get("embryo", ""),
            "isEmbryo": emb.get("isEmbryo", False),
            "image": emb.get("image", ""),
            "score": 0,  # Default score when processing fails
            "pgt": emb.get("pgt", "")
        })
    return ranked


def complete_ranking_pipeline(ranked_embryos: List[Dict]) -> List[Dict]:
    """
    Complete the ranking pipeline (sorting, lettering, reduction).
    
    Args:
        ranked_embryos: List of already ranked embryos
    
    Returns:
        list: Final processed list
    """
    # Sort by score
    sorted_embryos = sorted(ranked_embryos, key=lambda x: x['score'], reverse=True)
    
    # Add letters
    for i, embryo in enumerate(sorted_embryos):
        embryo['letter'] = ALPHABET[i] if i < len(ALPHABET) else f"Z{i - 25}"
    
    # Reduce to essential info
    result = keep_essential_info(sorted_embryos)
    print("[INFO] Pipeline complete. Ranked embryos:")
    for emb in result:
        print(emb)
    
    return result


def keep_essential_info(embryos_list: List[Dict]) -> List[Dict]:
    """
    Reduce embryo info to essential fields for database update.

    Args:
        embryos_list (list): Full list with all computed fields.

    Returns:
        list: Clean list with minimal fields.
    """
    return [
        {
            "embryo": emb["embryo"],
            "isEmbryo": emb["isEmbryo"],
            "image": emb["image"],
            "score": emb["score"],
            "letter": emb["letter"],
            "pgt": emb.get("pgt", "")
        }
        for emb in embryos_list
    ]