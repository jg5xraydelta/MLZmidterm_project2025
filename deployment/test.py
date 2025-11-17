import requests

url = 'http://localhost:9696/predict'

patient = {
  "age": 43,
  "sex": "m",
  "chestpaintype": "asy",
  "restingbp": 120,
  "cholesterol": 177,
  "fastingbs": 0,
  "restingecg": "lvh",
  "maxhr": 120,
  "exerciseangina": "y",
  "oldpeak": 2.5,
  "st_slope": "flat"
}

response = requests.post(url, json=patient)
predictions = response.json()

print(predictions)
if predictions['heart_failure']:
    print('patient most likely has heart disease')
else:
    print('patient is not likely to have heart disease')