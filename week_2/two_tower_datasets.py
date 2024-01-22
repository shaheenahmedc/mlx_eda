import torch
import pandas as pd
import random

class W2VData(torch.utils.data.Dataset):
    def __init__(self, corpus, window_size=2):
        self.corpus = corpus
        self.data = []
        self.create_data(window_size)

    def create_data(self, window_size):
        for index, row in self.corpus.iterrows():
            print (index/len(self.corpus))
            tokens = row['tokenized_ids']
            for i, target in enumerate(tokens):
                context = (
                    tokens[max(0, i - window_size) : i]
                    + tokens[i + 1 : i + window_size + 1]
                )
                if len(context) != 2 * window_size:
                    continue
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

# def two_tower_dataset(dataframe):
      # Doesn't create negative samples,
      # as positive and negative labels arise from users
#     new_rows = []
#     # Iterate through rows and expand
#     for index, row in dataframe.iterrows():
#         print (index/len(dataframe))
#         query = row['query']
#         for is_selected, passage in zip(row['passages']['is_selected'], row['passages']['passage_text']):
#             new_rows.append({
#                 'query': query,
#                 'is_selected': 1, # If passage selected by bing, log as 1
#                 'passage_text': passage
#             })
#             # Grab a random document, which Bing didn't return for this query
#             random_passage =
#             new_rows.append({
#                 'query': query,
#                 'is_selected': 0, # If passage not selected by bing, log as 0
#                 'passage_text': random_passage
#             })
#     # Convert the list of dictionaries to a DataFrame
#     result_df = pd.DataFrame(new_rows)
#     return result_df

def two_tower_dataset_optimized(dataframe):
    new_rows = []

    # Create a dictionary mapping queries to lists of passages
    query_passage_map = {}
    for index, row in dataframe.iterrows():
        query = row['query']
        if query not in query_passage_map:
            query_passage_map[query] = []
        query_passage_map[query].extend(row['passages']['passage_text'])

    all_queries = list(query_passage_map.keys())

    # Iterate through rows and expand
    for index, row in dataframe.iterrows():
        print(index / len(dataframe))
        query = row['query']
        for is_selected, passage in zip(row['passages']['is_selected'], row['passages']['passage_text']):
            # Add the passage selected by Bing
            new_rows.append({
                'query': query,
                'is_selected': 1,  # If passage selected by bing, log as 1
                'passage_text': passage
            })

            # Grab a random document not returned by Bing for this query
            while True:
                random_query = random.choice(all_queries)
                if random_query != query:
                    random_passage = random.choice(query_passage_map[random_query])
                    break

            # Add the randomly selected passage
            new_rows.append({
                'query': query,
                'is_selected': 0,  # If passage not selected by bing, log as 0
                'passage_text': random_passage
            })

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(new_rows)
    return result_df


class TwoTowerData(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        query = self.dataframe.iloc[index]['query']
        is_selected = self.dataframe.iloc[index]['is_selected']
        passage_text = self.dataframe.iloc[index]['passage_text']


        return torch.tensor(query), torch.tensor(is_selected), torch.tensor(passage_text)


def pad_sequence(sequences, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    padded_sequences = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, sequence in enumerate(sequences):
        length = sequence.size(0)
        padded_sequences[i, :length, ...] = sequence
    return padded_sequences

def collate_fn(batch):
    queries = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sentences = [item[2] for item in batch]

    return pad_sequence(queries), torch.stack(labels), pad_sequence(sentences)