import torch
import torch.nn.functional as F
import sentencepiece as spm

# Load the trained SentencePiece model


# Assuming your tokenizer and preprocessing method:
def tokenize_and_tensorize(text):
    sp = spm.SentencePieceProcessor()
    tokenizer = sp.Load("mymodel.model")
    # You might need to adjust this based on your tokenizer and preprocessing
    tokens = tokenizer.EncodeAsIds(str(text))
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension


def get_query_embedding(query):
    model = torch.load("two_tower.pth")
    query_tensor = tokenize_and_tensorize(query)
    with torch.no_grad():
        query_embedding, _ = model(
            query_tensor, query_tensor
        )  # We're only interested in the query's embedding
    return query_embedding


def compute_similarities(query_embedding, offline_sentence_embeddings_dict):
    similarities_dict = {}
    with torch.no_grad():
        i = 0
        for sentence, sentence_embedding in offline_sentence_embeddings_dict.items():
            cosine_similarity = F.cosine_similarity(query_embedding, sentence_embedding)
            similarities_dict[sentence] = cosine_similarity.item()
    return similarities_dict


def return_top_n_results(similarities_dict):
    sorted_indices = sorted(
        similarities_dict.items(), key=lambda item: item[1], reverse=True
    )
    top_matches = sorted_indices[:10]
    for i in top_matches:
        print(i)
