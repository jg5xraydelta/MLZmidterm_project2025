import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import xgboost as xgb
from typing import Dict, Any

from pydantic import BaseModel, Field
from typing import Literal


class Patient(BaseModel):
    # Numeric fields with observed ranges
    age: int = Field(..., ge=28, le=77)
    restingbp: int = Field(..., ge=0, le=200)
    cholesterol: int = Field(..., ge=0, le=603)
    fastingbs: Literal[0, 1]
    maxhr: int = Field(..., ge=60, le=202)
    oldpeak: float = Field(..., ge=-2.6, le=6.2)

    # Categorical fields from your counts
    sex: Literal["m", "f"]
    chestpaintype: Literal["asy", "nap", "ata", "ta"]
    restingecg: Literal["normal", "lvh", "st"]
    exerciseangina: Literal["n", "y"]
    st_slope: Literal["flat", "up", "down"]

# response

class PredictResponse(BaseModel):
    probability: float
    heart_failure: bool



app = FastAPI(title="heart-predict")

with open('model_xgb.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict_single(patient, dv, model):
    X = dv.transform([patient])
    y_pred = model.inplace_predict(X)
    return float(y_pred)

@app.post("/predict")
def predict(patient: Patient) -> PredictResponse:
    y_pred = predict_single(patient.dict(), dv, model)
    return PredictResponse(
        probability=y_pred,
        heart_failure=bool(y_pred >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)