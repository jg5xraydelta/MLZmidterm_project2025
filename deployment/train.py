
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle

output_file = f'model_xgb.bin'

df = pd.read_csv('heart.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


categorical = ['sex', 'chestpaintype', 'restingecg', 'exerciseangina',  'st_slope']

numerical = ['age','restingbp', 'cholesterol', 'fastingbs','maxhr','oldpeak']

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train_full.heartdisease.values
y_test = df_test.heartdisease.values

del df_train_full['heartdisease']
del df_test['heartdisease']


def train(df, y):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X_train_full = dv.transform(cat)

    dtrain_full = xgb.DMatrix(X_train_full, label=y_train, feature_names=dv.feature_names_)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 11,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
    }

    model = xgb.train(xgb_params, dtrain_full,
                  num_boost_round=70)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)
    dtest = xgb.DMatrix(X, label=y_test, feature_names=dv.feature_names_)

    y_pred = model.predict(dtest)

    return y_pred


dv, model = train(df_train_full, y_train)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


