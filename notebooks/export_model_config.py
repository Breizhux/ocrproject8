"""
À copier-coller dans une NOUVELLE cellule de 4_modelisation.ipynb,
APRÈS la cellule qui calcule `features_95` et `optimal_threshold`,
et à exécuter une seule fois pour générer le vrai model_config.json.
"""
import json

config = {
    "mlflow_model_name": "credit_scoring_lgbm",
    "mlflow_model_version": "2",          # version du modèle allégé (95 features)
    "mlflow_tracking_uri": "sqlite:///mlflow.db",
    "optimal_threshold": float(optimal_threshold),
    "features": list(features_95),
}

with open("../credit-scoring-api/src/model_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✓ Config exportée : {len(config['features'])} features, seuil={config['optimal_threshold']:.3f}")
