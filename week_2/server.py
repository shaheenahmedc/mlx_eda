import fastapi
import model

app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.inference_model = model.InferenceModel
  app.state.inference_model.eval()


@app.on_event("shutdown")
async def shutdown_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/learn_to_search")
async def on_language_challenge(request: fastapi.Request):

  # The POST request body has a text filed,
  # take it and tokenize it. Then feed it to
  # the language model and return the result.
  text = (await request.json())["text"]
  text = app.state.inference_model(text)
  return text
