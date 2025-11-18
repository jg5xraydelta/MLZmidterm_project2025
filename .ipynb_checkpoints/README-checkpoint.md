# MLZmidterm_project2025
heart failure prediction midterm project by Jared Gavin

# For a project, you repository/folder should contain the following:
## README.md with *Description of the problem *Instructions on how to run the project

PROBLEM STATEMENT:
The data set, heart.csv, contains the columns below.  The features are the first 11 columns and the 12th target column labeled 'heart disease' indicates whether or not the patient went on to develop heart disease.  Many of these features are common stats and tests that can be performed during a routine health check up.  Predicting heart disease early gives healthcare providers many options for reversing or slowing down the advancement of the disease.  A quick google search indicates that stats such as blood pressure, cholesterol, blood sugar and a test called and electrocardiogram are highly predictive of heart disease at certain thresholds.
One of the goals of this project is to determine which features included in the data demonstrate the most importance when predicting the target.  I will build 3 different models using logistic regression (LR), random forrest regressor (RFR) and xgboost (XGB).  After parameter tuning, I will compare these models with accuracy score and auc-roc metrics.  The best model will be chosen and a web service will be built that will take the features below as parameters.  The web service will produce a probability for developing heart disease and whether it is true or false that the patient is likely to develop heart disease.

Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
Chest Pain Type: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
Resting BP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
Fasting BS: fasting blood sugar [1: if Fasting BS > 120 mg/dl, 0: otherwise]
Resting ECG: resting electrocardiogram results
[Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression
of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria].
Max HR: maximum heart rate achieved [Numeric value between 60 and 202]
Exercise Angina: exercise-induced angina [Y: Yes, N: No]
Old peak: old peak = ST [Numeric value measured in depression]
ST _Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: down sloping]
Heart Disease: output class [1: heart disease, 0: Normal]



## Data -You should either commit the dataset you used or have clear instructions how to download the dataset
## Notebook (suggested name - notebook.ipynb) with *Data preparation and data cleaning *EDA, feature importance analysis
## *Model selection process and parameter tuning
The data set was found here: https://www.kaggle.com/datasets/tan5577/heart-failure-dataset.  It is included in the repo as 'heart.csv'.
The data was searched for missing data and there was none.  Spaces were replaced with underscores and all strings were lowercased.  The columns were divided into categorical and numerical columns.  EDA was performed with mean based risk, mutual info score, correlation and heart disease grouping.  
* The characteristics with the highest risk of heart disease based on the mean are being male, asymptomatic chest pain, having ST ecg abnormalities, having an exercise angina, and a down st slope.
* ST slope, chest pain type, and exercise angina have the highest mutual information scores.
* Age, old peak and fasting blood sugar have high positive correlation with heart disease.
* Patients with heart disease have a mean age 5 years older, higher blood pressure, lower cholesterol, higher blood sugar, lower max heart rate and high st peak depression.

Ultimately, XGBoost was chosen for the model being that it had the 2/3 better scores among accuracy score, kfold mean auc on validation data, and auc-roc on the full train vs test data.  Logistic Regression was a very close option that was very accurate.  Parameter tuning was performed for all 3 models. 
* LR - I tuned the C parameter and the threshold using auc scores and accuracy, respectively.
* RFR - I tuned the estimators, max depth and min samples leaf parameters using root mean square errors.
* XGB - I tuned the learning rate (eta), max depth and min child weight using auc scores.

## Script train.py (suggested name) *Training the final model *Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)





## Script predict.py (suggested name) *Loading the model -*Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)






## Files with dependencies *Pipenv and Pipenv.lock if you use Pipenv or equivalents: conda environment file, requirements.txt or pyproject.toml
## *Dockerfile for running the service






## Deployment -URL to the service you deployed or Video or image of how you interact with the deployed service