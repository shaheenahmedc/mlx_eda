import torch
from torch import nn

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        out = self.linear(embeds)
        log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return log_probs

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size):
        super(TwoTowerModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        # LSTM layers for query and sentence
        self.query_lstm = nn.LSTM(embedding_matrix.size(1), hidden_size, batch_first=True)
        self.sentence_lstm = nn.LSTM(embedding_matrix.size(1), hidden_size, batch_first=True)

        # Dense layer to produce final embeddings
        self.query_dense = nn.Linear(hidden_size, output_size)
        self.sentence_dense = nn.Linear(hidden_size, output_size)

    def forward(self, query, sentence):
        query_embed = self.embedding(query)
        sentence_embed = self.embedding(sentence)

        _, (query_hidden, _) = self.query_lstm(query_embed)
        _, (sentence_hidden, _) = self.sentence_lstm(sentence_embed)

        query_vector = self.query_dense(query_hidden.squeeze(0))
        sentence_vector = self.sentence_dense(sentence_hidden.squeeze(0))

        return query_vector, sentence_vector