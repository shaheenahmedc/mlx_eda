import gradio as gr
import torch
from model import Config
from model import DemoTransformer, TransformerSampler, Config
from transformer_lens import HookedTransformer

def return_transformer_output(user_input):
    reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    tokenizer = reference_gpt2.tokenizer

    model_cfg = Config(
        d_model=256,
        n_heads=4,
        d_head=64,
        d_mlp=1024,
        n_layers=2,
        n_ctx=256,
        d_vocab= 50257)
    sampling_model = DemoTransformer(model_cfg).to(model_cfg.device)
    sampling_model.load_state_dict(torch.load("gpt2_style_model_weights.pth"))

    sampler = TransformerSampler(sampling_model, tokenizer, model_cfg)

    prompt = 'Harry and Sally went to the mall '
    output = sampler.sample(prompt = prompt)



    return output

demo = gr.Interface(fn=return_transformer_output, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(show_api=False, share = True, server_name = '0.0.0.0', server_port=8092)
