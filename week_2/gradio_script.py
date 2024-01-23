import gradio as gr
import torch
from inference import get_query_embedding, compute_similarities
from model import CBOW, TwoTowerModel
from config import Config
import sentencepiece as spm

def return_top_n_docs(user_input):

    # Load offline document embeddings
    offline_embeddings_dict = torch.load(f'offline_embeddings_dict.json')

    # Load model
    cbow = CBOW(Config.SP_VOCAB_SIZE, Config.W2V_EMBEDDING_DIM)
    cbow.load_state_dict(torch.load(f'cbow_final_epoch.pt'))
    embedding_weights = cbow.embeddings.weight.data.detach()
    model = TwoTowerModel(
        embedding_matrix=torch.tensor(embedding_weights),
        hidden_size=Config.TWO_TOWER_HIDDEN_DIM,
        output_size=Config.TWO_TOWER_OUTPUT_DIM)
    model.load_state_dict(torch.load("path_to_two_tower_weights"), strict=False)
    model.eval()

    # Load the trained SentencePiece model
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('mymodel.model')

    # Get user query embedding
    query_embedding = get_query_embedding(user_input, model, tokenizer)

    # Compute similarities
    similarities = compute_similarities(query_embedding, offline_embeddings_dict, model, tokenizer)

    # Get top 10 matches (adjust as needed)
    sorted_indices = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_matches = sorted_indices[:10]

    return top_matches

demo = gr.Interface(fn=return_top_n_docs, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(show_api=False, share = False, server_name = '0.0.0.0', server_port=8091)
