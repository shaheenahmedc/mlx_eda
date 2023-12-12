import gradio as gr
import torch
from model import Language
from tokenizer import Tokenizer


def predict_language(user_input):
    # Load pretrained classification model
    state_dict_classification = torch.load(f'classification_model_epoch_4.pt')
    # Initialise tokenizer with vocab
    tokenizer = (Tokenizer()).load_vocab("./vocab.txt")
    # Use tokenizer vocab size to initialise language classification model
    classification_model = Language(torch.rand(len(tokenizer.vocab), 50), 7)
    # Load pre-trained weights into classification model
    classification_model.load_state_dict(state_dict_classification)
    # Set language ordering
    langs = ["German", "Esperanto", "French", "Italian", "Spanish", "Turkish", "English"]
    classification_model.eval()


    text = 'the cat sat on the hat'
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    predictions = classification_model(tokens)
    predictions_softmaxed = torch.nn.functional.softmax(predictions, dim=1)
    predictions_softmaxed = predictions_softmaxed.squeeze(0).tolist()

    result = [{"class": class_name, "value": value} for class_name, value in zip(langs, predictions_softmaxed)]
    return result

demo = gr.Interface(fn=predict_language, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(show_api=False, share = True)
