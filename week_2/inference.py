import torch
from torch.nn import functional as F

# Assuming your tokenizer and preprocessing method:
def tokenize_and_tensorize(text, tokenizer):
    # You might need to adjust this based on your tokenizer and preprocessing
    tokens = tokenizer.EncodeAsIds(str(text))
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

def get_query_embedding(query, model, tokenizer):
    query_tensor = tokenize_and_tensorize(query, tokenizer)
    with torch.no_grad():
        query_embedding, _ = model(query_tensor, query_tensor)  # We're only interested in the query's embedding
    return query_embedding

def create_offline_sentence_embeddings(sentences, model, tokenizer):
    embeddings_dict = {}
    with torch.no_grad():
        i=0
        for sentence in sentences:
            i+=1
            print (i/len(sentences))
            sentence_tensor = tokenize_and_tensorize(sentence, tokenizer)
            # Using dummy tensor for query since we only want the sentence embedding
            _, sentence_embedding = model(sentence_tensor, sentence_tensor)
            # Compute cosine similarity
            embeddings_dict[sentence] = sentence_embedding
    return embeddings_dict

def compute_similarities(query_embedding, offline_sentence_embeddings_dict, model, tokenizer):
    similarities_dict = {}
    with torch.no_grad():
        i=0
        for sentence, sentence_embedding in offline_sentence_embeddings_dict.items():
#             i+=1
#             print (i/len(sentences))
#             sentence_tensor = tokenize_and_tensorize(sentence, tokenizer)
            # Using dummy tensor for query since we only want the sentence embedding
#             _, sentence_embedding = model(sentence_tensor, sentence_tensor)
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(query_embedding, sentence_embedding)
            similarities_dict[sentence] = cosine_similarity.item()
    return similarities_dict