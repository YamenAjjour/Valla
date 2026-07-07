import os
import argparse
import logging
from datasets import load_dataset, load_from_disk
from valla.utils.dataset_utils import write_av_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def process_gerav_dataset(dataset_path: str):
    """
    Loads and processes the Gerav-style dataset, which may have pre-defined splits.

    Args:
        dataset_path (str): Path to the dataset (can be a local path or a dataset name from Hugging Face Hub).

    Returns:
        dict: A dictionary where keys are split names ('train', 'validation', 'test')
              and values are lists of processed data points in [id, label, text1, text2] format.
    """
    logging.info(f"Loading dataset from {dataset_path}")
    if os.path.isdir(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset(dataset_path, use_auth_token=True)

    processed_splits = {}
    for split_name, dataset_split in ds.items():
        problem_id = 0
        processed_data = []
        for example in tqdm(dataset_split, desc=f"Processing {split_name} split"):
            text1 = example.get("post_a", {}).get("text", "")
            text2 = example.get("post_b", {}).get("text", "")
            label = example.get("label")
            if text1 and text2 and label is not None:
                processed_data.append([problem_id, label, text1, text2])
                problem_id += 1
        processed_splits[split_name] = processed_data

    return processed_splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Gerav dataset and convert to AV CSV format.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path or name of the Gerav dataset (Hugging Face Hub or local).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output CSV files.')
    parser.add_argument('--dataset_name', type=str, default='gerav', help='Name of the dataset for output files.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the dataset splits
    processed_splits = process_gerav_dataset(args.dataset_path)

    # Map standard split names to our output file names
    split_map = {
        'train': 'train',
        'validation': 'val',
        'test': 'test'
    }

    for split_name, data in processed_splits.items():
        if split_name not in split_map:
            logging.warning(f"Unrecognized split '{split_name}', skipping.")
            continue

        output_suffix = split_map[split_name]
        output_path = os.path.join(args.output_dir, f'{args.dataset_name}_AV_{output_suffix}.csv')

        logging.info(f"{split_name.capitalize()} samples: {len(data)}")
        logging.info(f"Writing {len(data)} {split_name} samples to {output_path}")
        write_av_dataset(data, output_path)

    logging.info("Processing complete.")

    logging.warning(
        "Note: This script processes datasets that provide text pairs and a same/different author label. "
        "Therefore, it is not possible to generate Authorship Attribution (AA) specific splits or summaries "
        "that rely on author-text mappings, as done for some other datasets like PAN20. "
        "Only Authorship Verification (AV) splits have been created."
    )