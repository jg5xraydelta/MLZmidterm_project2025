import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import xgboost as xgb

app = FastAPI(title="heart-predict")

with open('model_xgb.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict_single(patient, dv, model):
    X = dv.transform([patient])
    y_pred = model.inplace_predict(X)
    return y_pre

@app.post("/predict")
def predict(patient):
    y_pred = predict_single(patient, dv, model)
    return ypred

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)