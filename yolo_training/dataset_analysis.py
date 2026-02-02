import os
import json
from collections import defaultdict
from itertools import combinations
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml


ROOT_DIR = "datasets"
YAML_PATH = "config/sperm-seg-new_datasets.yaml"
OUTPUT_DIR = "output_analysis"
CLASS_NAMES = {
    0: "sperm",
    1: "needle",
    2: "defect",
    3: "other",
    4: "egg",
    5: "pipette"
}


def count_instances_per_dataset(root_dir: str) -> Dict[str, Dict[int, int]]:
    """
    Count the number of class instances for each dataset in the root directory.

    Args:
        root_dir (str): Path to the root directory containing multiple datasets.

    Returns:
        Dict[str, Dict[int, int]]: A dictionary where the key is the dataset name and 
        the value is another dictionary mapping class IDs to their respective instance counts.
    """
    dataset_class_counts = {}
    for dataset_name in os.listdir(root_dir):
        labels_path = os.path.join(root_dir, dataset_name, "labels")
        if not os.path.isdir(labels_path):
            continue

        class_counts = defaultdict(int)
        for file in os.listdir(labels_path):
            if file.endswith(".txt"):
                with open(os.path.join(labels_path, file), "r") as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
        dataset_class_counts[dataset_name] = dict(class_counts)
    return dataset_class_counts


def compute_total_per_class(dataset_class_counts: Dict[str, Dict[int, int]]) -> Dict[int, int]:
    """
    Compute total class instances across all datasets.

    Args:
        dataset_class_counts (Dict[str, Dict[int, int]]): Dictionary with per-dataset class counts.

    Returns:
        Dict[int, int]: Total count of each class across all datasets.
    """
    total_counts = defaultdict(int)
    for dataset_counts in dataset_class_counts.values():
        for class_id, count in dataset_counts.items():
            total_counts[class_id] += count
    return dict(total_counts)


def propose_split(
    dataset_class_counts: Dict[str, Dict[int, int]],
    focus_class: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Dict[str, List[str]]:
    """
    Propose an optimal dataset split (train/val/test) based on balancing a specific class.

    Args:
        dataset_class_counts (Dict[str, Dict[int, int]]): Dictionary with per-dataset class counts.
        focus_class (int, optional): Class ID to balance across splits. Defaults to 0.
        train_ratio (float, optional): Proportion of focus class instances for training. Defaults to 0.6.
        val_ratio (float, optional): Proportion of focus class instances for validation. Defaults to 0.2.

    Returns:
        Dict[str, List[str]]: A dictionary with dataset names assigned to 'train', 'val', and 'test'.
    """
    datasets = list(dataset_class_counts.keys())
    total = sum([counts.get(focus_class, 0) for counts in dataset_class_counts.values()])
    target_train = total * train_ratio
    target_val = total * val_ratio

    best_split = None
    min_diff = float("inf")

    for i in range(1, len(datasets)-1):
        for train_combo in combinations(datasets, i):
            remaining = [d for d in datasets if d not in train_combo]
            for j in range(1, len(remaining)):
                val_combo = remaining[:j]
                test_combo = remaining[j:]

                train_sum = sum(dataset_class_counts[d].get(focus_class, 0) for d in train_combo)
                val_sum = sum(dataset_class_counts[d].get(focus_class, 0) for d in val_combo)

                diff = abs(train_sum - target_train) + abs(val_sum - target_val)
                if diff < min_diff:
                    min_diff = diff
                    best_split = {
                        "train": list(train_combo),
                        "val": list(val_combo),
                        "test": list(test_combo)
                    }
    return best_split


def update_yaml(yaml_path: str, split: Dict[str, List[str]], root: str) -> None:
    """
    Update a YAML configuration file with image paths for each split.

    Args:
        yaml_path (str): Path to the YAML config file.
        split (Dict[str, List[str]]): Dictionary containing dataset splits.
        root (str): Root path where datasets are stored.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    for key in ["train", "val", "test"]:
        data[key] = [os.path.join(root, ds, "images") for ds in split[key]]

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def plot_class_distribution(dataset_class_counts: Dict[str, Dict[int, int]], split: Dict[str, List[str]]) -> None:
    """
    Plot and save bar charts for class distribution before and after the split.

    Args:
        dataset_class_counts (Dict[str, Dict[int, int]]): Dictionary of class counts per dataset.
        split (Dict[str, List[str]]): Dataset split information.

    Returns:
        Dict[str, Dict[int, int]]: Dictionary of class distributions per split.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_class_counts = compute_total_per_class(dataset_class_counts)
    split_counts = {key: defaultdict(int) for key in split}

    for split_type, datasets in split.items():
        for ds in datasets:
            for class_id, count in dataset_class_counts[ds].items():
                split_counts[split_type][class_id] += count

    # Total distribution plot
    plt.figure(figsize=(10, 6))
    plt.bar([CLASS_NAMES[cid] for cid in total_class_counts], total_class_counts.values())
    plt.title("Distribución total por clase (antes del split)")
    plt.xlabel("Clases")
    plt.ylabel("Número de instancias")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "total_distribution.png"))
    plt.close()

    # Split distribution plots
    for split_type in ["train", "val", "test"]:
        plt.figure(figsize=(10, 6))
        class_ids = split_counts[split_type].keys()
        plt.bar([CLASS_NAMES[cid] for cid in class_ids], split_counts[split_type].values())
        plt.title(f"Distribución por clase en {split_type}")
        plt.xlabel("Clases")
        plt.ylabel("Número de instancias")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{split_type}_distribution.png"))
        plt.close()

    return split_counts


def save_analysis_json(
    dataset_class_counts: Dict[str, Dict[int, int]],
    split: Dict[str, List[str]],
    split_counts: Dict[str, Dict[int, int]]
) -> None:
    """
    Save a JSON file containing detailed information about dataset distributions and splits.

    Args:
        dataset_class_counts (Dict[str, Dict[int, int]]): Per-dataset class counts.
        split (Dict[str, List[str]]): Dataset split assignment.
        split_counts (Dict[str, Dict[int, int]]): Class distribution in each split.
    """
    analysis = {
        "dataset_counts": dataset_class_counts,
        "split_assignment": split,
        "split_class_distribution": {
            k: {CLASS_NAMES[cid]: count for cid, count in v.items()}
            for k, v in split_counts.items()
        }
    }
    with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(analysis, f, indent=4)


def main() -> None:
    dataset_class_counts = count_instances_per_dataset(ROOT_DIR)
    split = propose_split(dataset_class_counts, focus_class=0)
    update_yaml(YAML_PATH, split, root=ROOT_DIR)
    split_counts = plot_class_distribution(dataset_class_counts, split)
    save_analysis_json(dataset_class_counts, split, split_counts)


main()
