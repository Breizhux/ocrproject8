import json
import pandas as pd

with open("src/model_config.json") as f:
    config = json.load(f)
features = config["features"]

df = pd.read_pickle("/home/user/Projets/OCR-env/ocrproject8/credit-scoring-api/datas/app_test_final.pkl")

nan_counts = df[features].isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

print(f"Nombre de clients total : {len(df)}")
print(f"Features avec NaN parmi les 95 : {len(nan_counts)}")
print(nan_counts)
print()
print(f"Clients sans aucun NaN sur les 95 features : {df[features].dropna().shape[0]} / {len(df)}")
