import gradio as gr
import torch
from inference import get_query_embedding, compute_similarities


def return_top_n_docs(user_input):

    # Load offline document embeddings
    state_dict_classification = torch.load(f'offline_embeddings_dict.json')
    # Load model
    # How to store experiment variables like vocab_size?
    vocab_size = 1000
    cbow = CBOW(vocab_size, embedding_dim)
    cbow.load_state_dict(torch.load(f'cbow_final_epoch.pt'))
    embedding_weights = cbow.embeddings.weight.data.detach()
    model = TwoTowerModel(embedding_matrix=torch.tensor(embedding_weights), hidden_size=128, output_size=64)
    model.load_state_dict(torch.load("path_to_two_tower_weights"), strict=False)
    model.eval()
    # Get user query embedding
    tokenizer = get_query_embedding(query, model, tokenizer)
    # Use tokenizer vocab size to initialise language classification model
    classification_model = Language(torch.rand(len(tokenizer.vocab), 50), 7)
    # Load pre-trained weights into classification model
    classification_model.load_state_dict(state_dict_classification)
    # Set language ordering
    langs = ["German", "Esperanto", "French", "Italian", "Spanish", "Turkish", "English"]
    classification_model.eval()


    text = user_input
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    predictions = classification_model(tokens)
    predictions_softmaxed = torch.nn.functional.softmax(predictions, dim=1)
    predictions_softmaxed = predictions_softmaxed.squeeze(0).tolist()

    result = [{"class": class_name, "value": value} for class_name, value in zip(langs, predictions_softmaxed)]
    return result

demo = gr.Interface(fn=predict_language, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(show_api=False, share = False, server_name = '0.0.0.0', server_port=8090)
