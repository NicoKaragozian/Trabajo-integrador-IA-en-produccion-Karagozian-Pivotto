from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from datetime import date
import mlflow
import mlflow.xgboost
from feast import FeatureStore
import pandas as pd
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar pronósticos de producción de hidrocarburos.",
)

# --- Schemas de respuesta — corresponden a la spec OpenAPI del enunciado ---

class ForecastPoint(BaseModel):
    date: str
    prod: float

class ForecastResponse(BaseModel):
    id_well: str
    data: List[ForecastPoint]

class WellInfo(BaseModel):
    id_well: str

# --- Configuración ---
FEAST_REPO = os.getenv("FEAST_REPO_PATH", "/app/feature_store")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:6000")
MODEL_NAME = os.getenv("MODEL_NAME", "hydrocarbon-forecast-model")

FEAST_FEATURES = [
    "well_stats:avg_prod_pet_10m",
    "well_stats:avg_prod_gas_10m",
    "well_stats:last_prod_pet",
    "well_stats:n_readings",
    "well_stats:profundidad",
    "well_stats:tipo_extraccion",
]
FEATURE_COLS = [
    "avg_prod_pet_10m",
    "avg_prod_gas_10m",
    "last_prod_pet",
    "n_readings",
    "profundidad",
    "tipo_extraccion",
]

MODEL = None
STORE = None


@app.on_event("startup")
def startup():
    global MODEL, STORE
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        MODEL = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")
        logger.info(f"✅ Modelo '{MODEL_NAME}/Production' cargado")
    except Exception as e:
        logger.warning(
            f"⚠️ Modelo no disponible: {e}. Ejecutar training_pipeline primero."
        )
    STORE = FeatureStore(repo_path=FEAST_REPO)


# GET /api/v1/wells — parámetro: date_query
@app.get("/api/v1/wells", response_model=List[WellInfo])
def get_wells(
    date_query: date = Query(..., description="Fecha para la consulta (YYYY-MM-DD)")
):
    df = pd.read_parquet(f"{FEAST_REPO}/data/well_features.parquet")
    wells = df[df["fecha"] <= str(date_query)]["idpozo"].unique().tolist()
    return [{"id_well": str(w)} for w in sorted(wells)]


# GET /api/v1/forecast — parámetros: id_well, date_start, date_end
@app.get("/api/v1/forecast", response_model=ForecastResponse)
def get_forecast(
    id_well: str = Query(..., description="Identificador del pozo"),
    date_start: date = Query(..., description="Fecha de inicio (YYYY-MM-DD)"),
    date_end: date = Query(..., description="Fecha de fin (YYYY-MM-DD)"),
):
    if MODEL is None:
        raise HTTPException(
            503, "Modelo no disponible. Ejecutar training_pipeline.py primero."
        )

    # 1. Leer features del online store de Feast (baja latencia)
    online = STORE.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{"idpozo": id_well}],
    ).to_dict()

    if online.get("well_stats__avg_prod_pet_10m", [None])[0] is None:
        raise HTTPException(
            404, f"Pozo '{id_well}' no encontrado en el Feature Store."
        )

    # 2. Vector de features para el modelo (mismo orden que en training)
    X = pd.DataFrame(
        [
            {
                "avg_prod_pet_10m": online["well_stats__avg_prod_pet_10m"][0] or 0.0,
                "avg_prod_gas_10m": online["well_stats__avg_prod_gas_10m"][0] or 0.0,
                "last_prod_pet": online["well_stats__last_prod_pet"][0] or 0.0,
                "n_readings": online["well_stats__n_readings"][0] or 0,
                "profundidad": online["well_stats__profundidad"][0] or 0.0,
                "tipo_extraccion": online["well_stats__tipo_extraccion"][0] or 0,
            }
        ]
    )[FEATURE_COLS]

    # 3. Predicción base + decline curve mensual (configurable en config.yaml)
    base_pred = float(MODEL.predict(X)[0])
    with open("/app/config.yaml") as f:
        cfg = yaml.safe_load(f)
    decline = cfg.get("inference", {}).get("decline_rate_monthly", 0.02)

    months = pd.date_range(date_start, date_end, freq="MS")
    results = [
        {
            "date": str(m.date()),
            "prod": max(0.0, round(base_pred * ((1 - decline) ** i), 2)),
        }
        for i, m in enumerate(months)
    ]

    return {"id_well": id_well, "data": results}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}
