from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, field_validator
import pickle
import numpy as np

app = FastAPI()


## creating class for the request body
class PredictionRequest(BaseModel):
    features: conlist(float, min_length=20, max_length=20)

    @field_validator("features")
    @classmethod
    def check_range(cls, v):
        if any(item < 0 or item > 1 for item in v):
            raise ValueError("Each feature must be between 0 and 1")
        return v

# loading the scaler model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# loading the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


## prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
