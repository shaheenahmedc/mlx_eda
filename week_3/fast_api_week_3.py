from fastapi import FastAPI, Request
from pydantic import BaseModel

# import json
# import torch as t
# from model import DemoTransformer
# from model import TransformerSampler
# from transformer_lens import HookedTransformer

app = FastAPI()


@app.get("/")
def hello():
    return "ok"


# reference_gpt2 = HookedTransformer.from_pretrained(
#     "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
# )
# tokenizer = reference_gpt2.tokenizer


# class Config:
#     d_model: int = 768
#     #     debug: bool = True
#     layer_norm_eps: float = 1e-5
#     d_vocab: int = 50257
#     init_range: float = 0.02
#     n_ctx: int = 1024
#     d_head: int = 64
#     d_mlp: int = 3072
#     n_heads: int = 12
#     n_layers: int = 12


# model_cfg = Config(
#     d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=50257
# )

# # class InputData(BaseModel):
# #     text: str

# device = t.device("cuda" if t.cuda.is_available() else "cpu")

# sampling_model = DemoTransformer(model_cfg).to(device)
# model = sampling_model.load_state_dict(t.load("gpt2_style_model_weights.pth"))

# sampler = TransformerSampler(model, tokenizer)


# def return_story(prompt):
#     return sampler.sample(prompt, max_tokens_generated=80, top_k=5, frequency_penalty=5)


@app.post("/tell_me_stories")
async def on_tell_me_stories(request: Request):
    text = (await request.json())["text"]
    print("Input text:", text)
    # return {"story": return_story(text)}
    return {"story": "text"}
