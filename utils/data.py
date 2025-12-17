"""
Data loading and saving utilities.
"""

import os
import csv
import numpy as np
from typing import Union, Dict, List


def load_prompts(dir_path: str, file_name: str, allow_pickle: bool = False) -> np.ndarray:
    """Load prompts from numpy file."""
    try:
        return np.load(os.path.join(dir_path, file_name)).astype(np.int64)
    except ValueError as e:
        if "allow_pickle=False" in str(e):
            data = np.load(os.path.join(dir_path, file_name), allow_pickle=True)
            if data.dtype == np.dtype('O'):
                return np.array([np.array(x, dtype=np.int64) for x in data], dtype=np.int64)
            return data.astype(np.int64)
        raise


def write_array(file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Write numpy array to file with unique ID."""
    file_name = file_path.format(unique_id)
    np.save(file_name, array)


def write_guesses_to_csv(generations_per_prompt: int, 
                        generations_dict: Dict[str, np.ndarray], 
                        answers: np.ndarray, 
                        methods: List[str],
                        output_dir: str = "."):
    """Write guesses with ground truth labels to CSV files in specified directory."""
    for method in methods:
        filename = os.path.join(output_dir, f"guess_{method}_{generations_per_prompt}.csv")
        with open(filename, "w", newline='') as file_handle:
            print(f"Writing {filename}")
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess", "Ground Truth", "Is Correct"])

            for example_id in range(len(generations_dict[method])):
                guess = generations_dict[method][example_id]
                ground_truth = answers[example_id]
                is_correct = np.all(guess == ground_truth)
                
                row_output = [
                    example_id, 
                    str(list(guess)).replace(" ", ""),
                    str(list(ground_truth)).replace(" ", ""),
                    int(is_correct)
                ]
                writer.writerow(row_output)
