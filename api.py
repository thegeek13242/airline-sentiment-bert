from fastapi import FastAPI
import infer

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the API! Try /predict?text="}


@app.post("/predict")
async def predict(text: str):
    return {"prediction": infer.predict(text)}
