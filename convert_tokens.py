import argparse
import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def convert_tokens(args):
    print(f"Loading tokenizers...")
    neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    pythia_tokenizer = AutoTokenizer.from_pretrained(args.pythia_model)

    if pythia_tokenizer.pad_token is None:
        pythia_tokenizer.pad_token = pythia_tokenizer.eos_token

    os.makedirs(args.output_dir, exist_ok=True)

    files_to_convert = [
        "train_prefix.npy", 
        "train_dataset.npy", 
        "non_member_prefix.npy", 
        "member_prefix.npy",
        "val_dataset.npy",
        "val_prefix.npy"
    ]

    for filename in files_to_convert:
        input_path = os.path.join(args.input_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            continue
        
        print(f"Converting {filename}...")
        try:
            data = np.load(input_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Handle object arrays or regular arrays
        if data.dtype == np.dtype('O'):
            data_list = [x for x in data]
        else:
            data_list = [x for x in data]

        converted_data = []
        
        batch_size = 100
        for i in tqdm(range(0, len(data_list), batch_size)):
            batch = data_list[i:i+batch_size]
            
            # Decode Neo tokens to text
            # Ensure batch is a list of arrays or list of lists
            if isinstance(batch[0], np.ndarray):
                batch = [b.tolist() for b in batch]
                
            texts = neo_tokenizer.batch_decode(batch, skip_special_tokens=True)
            
            # Encode text to Pythia tokens
            encodings = pythia_tokenizer(texts, add_special_tokens=False)
            
            for input_ids in encodings.input_ids:
                converted_data.append(np.array(input_ids, dtype=np.int64))

        # Save as object array because lengths might differ
        output_path = os.path.join(args.output_dir, filename)
        np.save(output_path, np.array(converted_data, dtype=object))
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets", help="Directory containing Neo tokenized data")
    parser.add_argument("--output_dir", type=str, default="datasets_pythia", help="Directory to save Pythia tokenized data")
    parser.add_argument("--pythia_model", type=str, default="EleutherAI/pythia-1.4b", help="Pythia model name")
    args = parser.parse_args()
    convert_tokens(args)
