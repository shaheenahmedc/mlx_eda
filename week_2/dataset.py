from datasets import load_dataset_builder, load_dataset
import pandas as pd
import csv
import torch
import string
import tqdm

def load_HF_data(dataset_name, split, version=None):
    dataset = load_dataset(dataset_name, version=version, split=split)
    df_train = pd.DataFrame(dataset)
    return df_train

def process_ms_marco(dataset, output_file):
    # Open a CSV file for writing
    df_train = load_HF_data('ms_marco', 'train', version='v1.1')
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 1. Iterate over the rows of the dataframe
        for _, row in df_train.iterrows():
            # 2. For each row, access the passages column and then the passage_text key
            passage_texts = row['passages']['passage_text']

            # 3. Write each string from every list to the CSV file
            for text in passage_texts:
                writer.writerow([text])