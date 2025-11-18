import pickle
import pandas as pd
import xgboost as xgb

model_file = 'model_xgb.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

patient = {'age': 43,
 'sex': 'm',
 'chestpaintype': 'asy',
 'restingbp': 120,
 'cholesterol': 177,
 'fastingbs': 0,
 'restingecg': 'lvh',
 'maxhr': 120,
 'exerciseangina': 'y',
 'oldpeak': 2.5,
 'st_slope': 'flat'}


def predict_single(patient, dv, model):
    X = dv.transform([patient])
    y_pred = model.inplace_predict(X)
    return y_pred


y_pred = predict_single(patient, dv, model)


print('input', patient)
print('heart fail probability', y_pred)


