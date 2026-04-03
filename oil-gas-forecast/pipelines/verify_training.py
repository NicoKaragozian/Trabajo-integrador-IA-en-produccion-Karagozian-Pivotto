"""
Script de verificación del Training Pipeline.

Verifica que:
1. El offline store de Feast tiene datos disponibles
2. El Point-in-Time Join funciona para una fecha de corte dada
3. El entrenamiento corre sin errores y loguea en MLflow
4. El modelo queda registrado en Production en el Registry

Uso:
    cd oil-gas-forecast
    python pipelines/verify_training.py --fecha 2024-06-01
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def verify_offline_store(repo_path: str = "./feature_store"):
    parquet = Path(repo_path) / "data" / "well_features.parquet"
    assert parquet.exists(), f"Offline store no encontrado: {parquet}"
    df = pd.read_parquet(parquet)
    assert not df.empty, "El parquet está vacío"
    logger.info(f"✅ Offline store: {len(df):,} filas, {df['idpozo'].nunique():,} pozos")
    logger.info(f"   Rango temporal: {df['fecha'].min()} → {df['fecha'].max()}")
    return df


def verify_training_data(fecha_corte: str, repo_path: str = "./feature_store"):
    from pipelines.training_pipeline import get_training_data

    training_df = get_training_data(fecha_corte, repo_path)
    assert not training_df.empty, "get_training_data devolvió un DataFrame vacío"
    required_cols = [
        "prod_pet",  # target
        "avg_prod_pet_10m", "avg_prod_gas_10m", "last_prod_pet",
        "n_readings", "profundidad", "tipo_extraccion",
    ]
    for col in required_cols:
        assert col in training_df.columns, f"Columna faltante en training data: {col}"
    logger.info(f"✅ Training data: {len(training_df)} pozos con {len(training_df.columns)} features")
    return training_df


def verify_mlflow_run(fecha_corte: str, config_path: str = "config.yaml"):
    from pipelines.training_pipeline import train

    logger.info(f"Lanzando entrenamiento de prueba con fecha_corte={fecha_corte}...")
    run_id, version = train(fecha_corte, config_path)
    assert run_id, "El run_id está vacío"
    assert version, "La versión del modelo está vacía"
    logger.info(f"✅ Run completado: {run_id}")
    logger.info(f"✅ Modelo en Production: versión {version}")
    return run_id, version


def verify_model_registry(model_name: str = "hydrocarbon-forecast-model"):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]

    assert production, f"No hay versiones en Production para '{model_name}'"
    assert len(production) == 1, f"Hay {len(production)} versiones en Production (debe ser exactamente 1)"
    logger.info(f"✅ Model Registry: '{model_name}' v{production[0].version} en Production")
    return production[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar el Training Pipeline")
    parser.add_argument("--fecha", default="2024-06-01", help="Fecha de corte YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Omitir el entrenamiento y la verificación del registry (solo verificar datos)",
    )
    args = parser.parse_args()

    repo_path = os.getenv("FEAST_REPO_PATH", "./feature_store")
    model_name = os.getenv("MODEL_NAME", "hydrocarbon-forecast-model")

    print("\n=== Verificación del Training Pipeline ===\n")

    verify_offline_store(repo_path)
    verify_training_data(args.fecha, repo_path)

    if not args.skip_train:
        verify_mlflow_run(args.fecha, args.config)

    verify_model_registry(model_name)
    print("\n✅ Todas las verificaciones pasaron correctamente.\n")
