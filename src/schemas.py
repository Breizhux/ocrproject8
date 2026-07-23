"""
Schémas de validation des données d'entrée/sortie de l'API (Pydantic).

Les 95 features du modèle étant toutes numériques (voir notebook de
modélisation : "Valeurs manquantes : 0" après feature engineering),
on valide un dictionnaire {feature: valeur float}, en s'assurant que :
- toutes les features attendues sont présentes (pas de champ manquant),
- aucune feature inconnue n'est envoyée,
- toutes les valeurs sont bien numériques (pas de texte),
- pas de NaN/Infini.
"""
import math
from typing import Dict

from pydantic import BaseModel, field_validator, model_validator

from .model_loader import get_features


class ClientData(BaseModel):
    """Données d'un client à scorer : dictionnaire feature -> valeur numérique."""

    features: Dict[str, float]

    @model_validator(mode="after")
    def check_features_complete_and_valid(self):
        expected = set(get_features())
        received = set(self.features.keys())

        missing = expected - received
        if missing:
            raise ValueError(
                f"Features manquantes ({len(missing)}) : {sorted(missing)[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

        unknown = received - expected
        if unknown:
            raise ValueError(
                f"Features inconnues du modèle : {sorted(unknown)[:10]}"
            )

        for name, value in self.features.items():
            if value is None:
                raise ValueError(f"La feature '{name}' ne peut pas être null.")
            if isinstance(value, bool):
                raise ValueError(f"La feature '{name}' doit être numérique, pas booléenne.")
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"La feature '{name}' contient une valeur NaN/Infinie.")

        return self


class PredictionResponse(BaseModel):
    """Réponse de l'API : probabilité de défaut + décision + métadonnées."""

    probability_default: float
    decision: str  # "ACCORDÉ" ou "REFUSÉ"
    threshold_used: float
    model_version: str
