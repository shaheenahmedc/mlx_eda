from fastapi import FastAPI, Request
from pydantic import BaseModel
import json

app = FastAPI()


@app.get("/")
def hello():
    return "ok"


class InputData(BaseModel):
    text: str


@app.post("/what_language_is_this")
def predict_using_input(input_data: InputData):
    print("besbsb1s", input_data)
    return [
        {"class": "German", "value": 0.1},
        {"class": "Esperanto", "value": 0.4},
        {"class": "French", "value": 0.1},
        {"class": "Italian", "value": 0.1},
        {"class": "Spanish", "value": 0.1},
        {"class": "Turkish", "value": 0.1},
        {"class": "English", "value": 0.1},
    ]
