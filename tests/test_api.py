"""
Tests unitaires de l'API.

On mocke le modèle (model_singleton) pour ne pas dépendre de MLflow/du
vrai modèle lors des tests CI. On couvre les points de vigilance des
consignes : données manquantes, valeurs hors plage, types incorrects.
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

FAKE_FEATURES = ["EXT_SOURCE_MEAN", "EMPLOYED_YEARS", "AMT_ANNUITY"]
FAKE_THRESHOLD = 0.52


@pytest.fixture
def client():
    """
    Fixture qui patch le chargement du modèle avant d'importer l'app,
    pour isoler les tests de toute dépendance MLflow réelle.
    """
    with patch("src.model_loader.load_config") as mock_config:
        mock_config.return_value = {
            "mlflow_model_name": "credit_scoring_lgbm",
            "mlflow_model_version": "2",
            "mlflow_tracking_uri": "sqlite:///mlflow.db",
            "optimal_threshold": FAKE_THRESHOLD,
            "features": FAKE_FEATURES,
        }
        with patch("mlflow.pyfunc.load_model") as mock_load_model:
            fake_model = MagicMock()
            fake_model.predict.return_value = [0.75]  # proba de défaut simulée
            mock_load_model.return_value = fake_model

            from src.api import app
            from src.model_loader import model_singleton
            model_singleton._model = None
            model_singleton._config = None

            with TestClient(app) as test_client:
                yield test_client


def valid_payload():
    return {"features": {f: 0.5 for f in FAKE_FEATURES}}


# --- Cas nominal ---

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_input(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["probability_default"] == 0.75
    assert body["decision"] == "REFUSÉ"  # 0.75 >= seuil 0.52
    assert body["threshold_used"] == FAKE_THRESHOLD


# --- Cas limites : données manquantes ---

def test_predict_missing_feature(client):
    payload = valid_payload()
    del payload["features"]["EMPLOYED_YEARS"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "manquantes" in str(response.json()).lower()


def test_predict_empty_features(client):
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422


# --- Cas limites : valeurs hors plage / invalides ---

def test_predict_nan_value(client):
    # NaN n'est pas un JSON valide (RFC 8259) : httpx refuse de l'encoder
    # via json=..., donc on simule un client "laxiste" envoyant du JSON
    # brut contenant le token NaN, que Python sait quand même parser.
    raw_body = (
        '{"features": {"EXT_SOURCE_MEAN": NaN, '
        '"EMPLOYED_YEARS": 0.5, "AMT_ANNUITY": 0.5}}'
    )
    response = client.post(
        "/predict",
        content=raw_body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_negative_extreme_value(client):
    # ex: un âge/anciennete négatif — le modèle doit quand même répondre,
    # la validation métier de plage n'est pas bloquante ici, seul le type compte.
    payload = valid_payload()
    payload["features"]["EMPLOYED_YEARS"] = -5.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# --- Cas limites : types incorrects ---

def test_predict_text_instead_of_number(client):
    payload = valid_payload()
    payload["features"]["AMT_ANNUITY"] = "beaucoup"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unknown_feature(client):
    payload = valid_payload()
    payload["features"]["FEATURE_INCONNUE"] = 1.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "inconnues" in str(response.json()).lower()
