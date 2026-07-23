"""
Chargement du modèle de scoring depuis le MLflow Model Registry.

Le modèle est chargé UNE SEULE FOIS (au démarrage de l'API via un singleton),
puis réutilisé pour toutes les requêtes. Voir consignes.md - étape 2,
point de vigilance "ne pas charger le modèle à chaque requête".
"""
import json
import os
import logging
from pathlib import Path

import mlflow.pyfunc

logger = logging.getLogger("model_loader")

CONFIG_PATH = Path(__file__).parent / "model_config.json"


def load_config() -> dict:
    """Charge la config (features, seuil, URI MLflow) depuis model_config.json."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if not config["features"]:
        raise RuntimeError(
            "model_config.json ne contient aucune feature. "
            "Exécutez notebooks/export_model_config.py depuis le notebook de modélisation."
        )
    return config


class ModelSingleton:
    """
    Garantit que le modèle n'est chargé qu'une seule fois en mémoire,
    même si plusieurs threads appellent get_model().
    """
    _instance = None
    _model = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._model is not None:
            return  # déjà chargé, on ne recharge pas

        self._config = load_config()

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", self._config["mlflow_tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = (
            f"models:/{self._config['mlflow_model_name']}/"
            f"{self._config['mlflow_model_version']}"
        )
        logger.info(f"Chargement du modèle depuis {model_uri} ...")
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modèle chargé en mémoire.")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def config(self):
        if self._config is None:
            self._config = load_config()
        return self._config


# Instance unique importée partout dans l'app
model_singleton = ModelSingleton()


def get_model():
    return model_singleton.model


def get_features() -> list:
    return model_singleton.config["features"]


def get_threshold() -> float:
    return model_singleton.config["optimal_threshold"]
