import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal_length':2, 'sepal_width':9, 'petal_length':6, 'petal_width':2})

print(r.json())