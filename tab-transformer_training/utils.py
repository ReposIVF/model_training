import pandas as pd
from typing import Optional

def merge_predictions_with_metadata(
    metadata_path: str,
    results_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Merge predictions and all columns from results.csv into image_metadata.csv based on 'image id'.

    Args:
        metadata_path (str): Path to image_metadata.csv.
        results_path (str): Path to results.csv (must have 'image id').
        output_path (Optional[str]): Path to save the merged CSV. If None, no file is saved.

    Returns:
        pd.DataFrame: Merged DataFrame containing metadata + results columns.
    """
    metadata_df = pd.read_csv(metadata_path)
    results_df = pd.read_csv(results_path)

    merged_df = metadata_df.merge(
        results_df,
        on="image id",
        how="left"
    )

    if output_path:
        merged_df.to_csv(output_path, index=False)

    return merged_df


merged = merge_predictions_with_metadata(
    metadata_path="data/image_metadata_30m_118-128-fixed.csv",
    results_path="results_timelapse_image_metadata_30m_118-128-fixed.csv",
    output_path="prediction_results_image_metadata_30m_118-128-19-10-2025-fixed.csv"
)
