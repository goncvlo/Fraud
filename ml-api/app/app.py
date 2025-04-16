from fastapi import FastAPI
from pydantic import BaseModel
from .inference import predict

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def read_root():
    return {"status": "API is live"}

@app.post("/predict")
def _predict(input_data: InputData):
    result = predict(input_data.feature1, input_data.feature2)
    return {"prediction": result}
