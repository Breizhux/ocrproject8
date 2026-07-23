"""
API de scoring crédit — Prêt à Dépenser.

- Le modèle est chargé une seule fois au démarrage (lifespan FastAPI).
- POST /predict : reçoit les données d'un client, retourne la probabilité
  de défaut et la décision (ACCORDÉ / REFUSÉ) selon le seuil optimal.
- Toutes les erreurs de validation renvoient un 422 explicite (géré nativement
  par Pydantic/FastAPI). Les erreurs internes renvoient un 500 avec message générique.
"""
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .model_loader import model_singleton, get_features, get_threshold
from .schemas import ClientData, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup : chargement du modèle UNE SEULE FOIS ---
    logger.info("Démarrage de l'API : chargement du modèle...")
    model_singleton.load()
    logger.info("Modèle prêt.")
    yield
    # --- Shutdown ---
    logger.info("Arrêt de l'API.")


app = FastAPI(
    title="Credit Scoring API — Prêt à Dépenser",
    description="API de scoring crédit pour le département Crédit Express",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"status": "ok", "service": "credit-scoring-api"}


@app.get("/health")
def health():
    """Endpoint de santé pour le monitoring / orchestrateur (Docker healthcheck, k8s...)."""
    try:
        _ = model_singleton.model  # vérifie que le modèle est bien chargé
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        logger.error(f"Healthcheck échoué : {e}")
        raise HTTPException(status_code=503, detail="Modèle non disponible")


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    """
    Prend les données d'un client (95 features) et retourne
    la probabilité de défaut ainsi que la décision.
    """
    start = time.perf_counter()

    try:
        features = get_features()
        # On respecte l'ordre exact des features attendu par le modèle
        row = pd.DataFrame([[client.features[f] for f in features]], columns=features)

        model = model_singleton.model
        proba = float(model.predict(row)[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")

    threshold = get_threshold()
    decision = "REFUSÉ" if proba >= threshold else "ACCORDÉ"
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"Prédiction effectuée en {elapsed_ms:.1f}ms | proba={proba:.4f} | décision={decision}"
    )

    return PredictionResponse(
        probability_default=proba,
        decision=decision,
        threshold_used=threshold,
        model_version=model_singleton.config["mlflow_model_version"],
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handler dédié : par défaut, FastAPI renvoie les erreurs brutes de Pydantic
    (exc.errors()), qui peuvent contenir :
    - l'exception Python elle-même dans "ctx.error" (non sérialisable en JSON),
    - la valeur brute envoyée par le client dans "input", qui peut être NaN/Inf
      (interdit par Starlette en JSON strict).
    On reconstruit donc un message d'erreur simple et toujours sérialisable,
    sans jamais réutiliser ces champs bruts.
    """
    errors = [
        {
            "loc": list(err.get("loc", [])),
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Exception non gérée : {exc}")
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur."})
