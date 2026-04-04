"""
Feature Pipeline — Producción de Pozos No Convencionales

Descarga datos crudos → limpieza → features MIT → materializa en Feast.
Puede correrse localmente sin Docker:
    cd oil-gas-forecast
    python pipelines/feature_pipeline.py
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
from feast import FeatureStore

# Permite importar feature_store/features.py desde cualquier CWD
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_store.features import pozo, well_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = (
    "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c"
    "/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725"
    "/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"
)
RAW_PATH = Path("data/raw/pozos.csv")
FEAST_REPO_PATH = Path("feature_store")


def load_raw_data() -> pd.DataFrame:
    if RAW_PATH.exists():
        logger.info(f"Cargando datos cacheados desde {RAW_PATH}")
        return pd.read_csv(RAW_PATH)
    logger.info("Descargando dataset desde datos.gob.ar...")
    df = pd.read_csv(DATA_URL)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    logger.info(f"Dataset guardado: {len(df):,} filas → {RAW_PATH}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.strip()

    # La columna de fecha en el dataset se llama fecha_data
    df["fecha"] = pd.to_datetime(df["fecha_data"])

    # Producción nunca puede ser negativa (hay valores -0.001 y -12.26 en el dataset)
    for col in ["prod_pet", "prod_gas", "prod_agua"]:
        df[col] = df[col].clip(lower=0).fillna(0)

    df = df.dropna(subset=["idpozo"])

    # Encodear tipo de extracción como entero (601 nulos → categoría 0 = "Sin sistema")
    tipos = {t: i + 1 for i, t in enumerate(sorted(df["tipoextraccion"].dropna().unique()))}
    df["tipo_extraccion"] = df["tipoextraccion"].map(tipos).fillna(0).astype(int)

    # Profundidad: rellenar con la media del pozo; si el pozo tampoco tiene datos, 0
    df["profundidad"] = (
        df.groupby("idpozo")["profundidad"]
        .transform(lambda x: x.fillna(x.mean()))
        .fillna(0)
        .astype(float)
    )

    return df


def compute_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Computa Model-Independent Transformations (MIT).

    Estas transformaciones no dependen del algoritmo de ML final y pueden
    ser reutilizadas por múltiples modelos. Se almacenan en el Feature Store
    para garantizar consistencia entre training e inferencia.
    """
    df = df.sort_values(["idpozo", "fecha"])
    grp = df.groupby("idpozo")

    # Promedio de las últimas `window` lecturas por pozo
    df["avg_prod_pet_10m"] = grp["prod_pet"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df["avg_prod_gas_10m"] = grp["prod_gas"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # shift(1): última lectura conocida — evita leakage (en producción no conocemos el valor actual)
    df["last_prod_pet"] = grp["prod_pet"].shift(1).fillna(0)

    # Cantidad de registros históricos disponibles para el pozo en ese momento
    df["n_readings"] = grp.cumcount() + 1

    feature_cols = [
        "idpozo", "fecha",
        "prod_pet", "prod_gas", "prod_agua",
        "avg_prod_pet_10m", "avg_prod_gas_10m",
        "last_prod_pet", "n_readings",
        "profundidad", "tipo_extraccion",
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available].reset_index(drop=True)


def materialize_to_feast(df: pd.DataFrame, repo_path: Path = FEAST_REPO_PATH):
    # 1. Offline store: parquet con el historial completo
    parquet_path = repo_path / "data" / "well_features.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Offline store guardado: {len(df):,} filas → {parquet_path}")

    # 2. Aplicar definiciones al registry (crea/actualiza registry.db)
    store = FeatureStore(repo_path=str(repo_path))
    store.apply([pozo, well_stats])
    logger.info("Registry de Feast actualizado")

    # 3. Online store: write_to_online_store con la última lectura por pozo
    # Usamos write_to_online_store (no materialize) por compatibilidad con la versión
    # de Feast/pandas instalada en el entorno (materialize falla con tz_convert).
    # Tomamos la fila más reciente por pozo — el estado actual para inferencia.
    latest = df.sort_values("fecha").groupby("idpozo").last().reset_index()

    # event_timestamp con timezone UTC: Feast lo requiere para el control de TTL
    latest["event_timestamp"] = pd.Timestamp.now(tz="UTC")

    online_cols = [
        "idpozo", "event_timestamp",
        "avg_prod_pet_10m", "avg_prod_gas_10m",
        "last_prod_pet", "n_readings",
        "profundidad", "tipo_extraccion",
    ]
    available = [c for c in online_cols if c in latest.columns]
    store.write_to_online_store(feature_view_name="well_stats", df=latest[available])

    n_wells = latest["idpozo"].nunique()
    logger.info(f"Online store actualizado: {n_wells:,} pozos activos")


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_features = compute_features(df_clean)
    materialize_to_feast(df_features)
    logger.info("✅ Feature Pipeline completado")
