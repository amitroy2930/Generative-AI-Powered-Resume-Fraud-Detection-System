import os
import sys
import pandas as pd
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer

def parse_arguments():
    """
    Parse and validate command-line arguments.
    Returns:
        tuple: Input folder path and output folder path.
    """
    if len(sys.argv) != 3:
        print("Usage: python data_extraction.py <input_folder> <output_folder>")
        sys.exit(1)
    return sys.argv[1], sys.argv[2]


def load_model():
    """
    Load the SentenceTransformer model.
    Returns:
        SentenceTransformer: Loaded model instance.
    """
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def process_file(file_path, model, output_folder):
    """
    Process a single CSV file to generate embeddings and save outputs.
    Args:
        file_path (str): Path to the CSV file.
        model (SentenceTransformer): Loaded SentenceTransformer model.
        output_folder (str): Folder where outputs will be saved.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    raw_text_df = pd.read_csv(file_path)

    # Skip files with less than 100 rows
    if len(raw_text_df) < 100:
        print(f"Skipping {file_name}, less than 100 entries.")
        return

    # Extract data
    index_to_user_map = raw_text_df['resource_id'].to_dict()
    raw_text_list = raw_text_df['clean_text'].tolist()

    print(f"Processing {file_name}: {len(raw_text_list)} entries.")

    # Generate embeddings
    start_time = time.time()
    embeddings = model.encode(raw_text_list)
    end_time = time.time()

    print(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")

    # Save embeddings and mappings
    np.save(os.path.join(output_folder, f'{file_name}_embeddings.npy'), embeddings)
    with open(os.path.join(output_folder, f'{file_name}_index_to_user_map.json'), 'w') as f:
        json.dump(index_to_user_map, f)

    print(f"Saved embeddings and mapping for {file_name}.")


def main():
    """
    Main function to process all CSV files in the input folder.
    """
    # Parse command-line arguments
    folder_path, output_folder = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load model
    model = load_model()

    # Process each CSV file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            process_file(file_path, model, output_folder)


if __name__ == "__main__":
    main()