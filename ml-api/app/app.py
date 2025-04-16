from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .inference import predict

app = FastAPI()

# CORS is needed here!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# input schema
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
