import csv
import sentencepiece as spm


def prepare_sentencepiece_dataset(dataframe, output_file = 'output.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 1. Iterate over the rows of the dataframe
        for _, row in dataframe.iterrows():
            # 2. For each row, access the passages column and then the passage_text key
            passage_texts = row['passages']['passage_text']

            # 3. Write each string from every list to the CSV file
            for text in passage_texts:
                writer.writerow([text])

def train_sentencepiece(input_file, model_prefix, vocab_size, character_coverage, model_type):

    train_args = {
        'input': input_file,             # Input file
        'model_prefix': model_prefix,        # Prefix for the output model files (.model and .vocab)
        'vocab_size': vocab_size,              # Size of the vocabulary
        'character_coverage': character_coverage,     # Character coverage to be considered for the model. Good defaults are: 0.9995 for languages with rich character sets like Japanese or Chinese and 0.9997 for others
        'model_type': model_type,          # Model type can be 'unigram' (default), 'bpe', 'char', or 'word'
        # Add other parameters as needed.
    }

    # Train the model
    spm.SentencePieceTrainer.Train(' '.join([f'--{k}={v}' for k, v in train_args.items()]))

    print("Model trained and saved as mymodel.model and mymodel.vocab!")
