"""
Training Pipeline — Producción de Pozos No Convencionales

Lee features del offline store de Feast → entrena XGBoost → loguea en MLflow
→ registra modelo en Production.

Uso:
    cd oil-gas-forecast
    python pipelines/training_pipeline.py --fecha 2024-06-01
    python pipelines/training_pipeline.py --fecha 2024-06-01 --config config.yaml
"""
import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from feast import FeatureStore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEAST_FEATURES = [
    "well_stats:prod_pet",  # target — incluido en el join para no perder el PIT alignment
    "well_stats:avg_prod_pet_10m",
    "well_stats:avg_prod_gas_10m",
    "well_stats:last_prod_pet",
    "well_stats:n_readings",
    "well_stats:profundidad",
    "well_stats:tipo_extraccion",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_training_data(fecha_corte: str, repo_path: str = "./feature_store") -> pd.DataFrame:
    """
    Lee features del offline store de Feast con Point-in-Time Join.
    Solo usa datos disponibles ANTES de fecha_corte → sin data leakage.
    """
    store = FeatureStore(repo_path=repo_path)

    df_all = pd.read_parquet(Path(repo_path) / "data" / "well_features.parquet")
    df_all["fecha"] = pd.to_datetime(df_all["fecha"])
    fecha_corte_ts = pd.to_datetime(fecha_corte)
    df_filtered = df_all[df_all["fecha"] <= fecha_corte_ts].copy()

    if df_filtered.empty:
        raise ValueError(f"No hay datos hasta {fecha_corte}")

    # Snapshot: última lectura por pozo hasta la fecha de corte
    entity_df = (
        df_filtered.sort_values("fecha")
        .groupby("idpozo")
        .tail(1)[["idpozo", "fecha"]]
        .rename(columns={"fecha": "event_timestamp"})
    )
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # Point-in-Time Join automático de Feast
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEAST_FEATURES,
    ).to_df()

    logger.info(f"Training set: {len(training_df)} pozos hasta {fecha_corte}")
    return training_df


def plot_feature_importance(model: xgb.XGBRegressor, run_id: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax, max_num_features=15, title="Feature Importance")
    path = f"/tmp/feature_importance_{run_id[:8]}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def register_model(run_id: str, model_name: str) -> str:
    """Registra el modelo y lo promueve a Production, archivando los anteriores."""
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/xgb_model"

    # Crear el registered model si no existe (primer entrenamiento)
    try:
        client.create_registered_model(model_name)
        logger.info(f"Registered Model '{model_name}' creado")
    except mlflow.exceptions.MlflowException:
        pass  # ya existe

    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

    # Archivar versiones viejas en Production para que solo haya una activa
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.current_stage == "Production" and v.version != mv.version:
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )

    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage="Production"
    )
    logger.info(f"Modelo v{mv.version} promovido a Production en el Registry")
    return mv.version


def train(fecha_corte: str, config_path: str = "config.yaml") -> tuple:
    config = load_config(config_path)
    model_cfg = config["model"]

    # Puerto 6000 — MLflow corre en :6000 según docker-compose.yml
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "hydrocarbon-forecast"))

    repo_path = os.getenv("FEAST_REPO_PATH", "./feature_store")
    df = get_training_data(fecha_corte, repo_path).dropna()

    X = df[model_cfg["features"]]
    y = df[model_cfg["target"]]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"xgb_{fecha_corte}") as run:

        # 1. Parámetros y metadata
        mlflow.log_params(model_cfg["xgb_params"])
        mlflow.log_param("fecha_corte", fecha_corte)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_pozos", df["idpozo"].nunique() if "idpozo" in df.columns else "?")
        mlflow.log_param("features", str(model_cfg["features"]))
        mlflow.log_param("target", model_cfg["target"])
        mlflow.set_tag("stage", "production-candidate")
        mlflow.set_tag("dataset", "pozos-no-convencionales-argentina")
        mlflow.set_tag("feature_store", "feast-local")

        # 2. Entrenar
        model = xgb.XGBRegressor(**model_cfg["xgb_params"])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 3. Métricas de validación
        preds_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
        # MAPE con suavizado para evitar división por cero en pozos con prod~0
        mape = float(np.mean(np.abs((y_val - preds_val) / (y_val + 1e-8))) * 100)

        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mape", mape)
        logger.info(f"Métricas: MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.1f}%")

        # 4. Artefactos
        fi_path = plot_feature_importance(model, run.info.run_id)
        mlflow.log_artifact(fi_path, "plots")
        mlflow.log_artifact(config_path, "config")  # trazabilidad del config usado

        # 5. Modelo serializado
        mlflow.xgboost.log_model(model, "xgb_model")

        # 6. Registrar en Model Registry y promover a Production
        model_name = os.getenv("MODEL_NAME", "hydrocarbon-forecast-model")
        version = register_model(run.info.run_id, model_name)

        return run.info.run_id, version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de pronóstico de producción")
    parser.add_argument("--fecha", required=True, help="Fecha de corte YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_id, version = train(args.fecha, args.config)
    print(f"✅ Run ID: {run_id}")
    print(f"✅ Modelo registrado como versión {version} en Production")
