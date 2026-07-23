import json
import requests
import pandas as pd

API_URL = "http://localhost:8000/predict"

with open("src/model_config.json") as f:
    config = json.load(f)
features = config["features"]

df = pd.read_pickle("datas/app_test_final.pkl")

for i in range(10):
    client_row = df.iloc[i]
    payload = {"features": {f: float(client_row[f]) for f in features}}
    response = requests.post(API_URL, json=payload)
    proba = response.json().get("probability_default")
    print(f"Client {i} : proba={proba}")
