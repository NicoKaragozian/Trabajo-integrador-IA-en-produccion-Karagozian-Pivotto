"""
Feature Pipeline — Producción de Pozos No Convencionales

Descarga datos crudos → limpieza → features MIT → materializa en Feast.
Puede correrse localmente sin Docker:
    cd oil-gas-forecast
    python pipelines/feature_pipeline.py

La configuración (rutas, URL del dataset, hiperparámetros) se lee desde
oil-gas-forecast/config.yaml para evitar hardcodear valores en el código.
"""
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import yaml
from feast import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = BASE_DIR / "config.yaml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Archivo de configuración no encontrado: {_CONFIG_PATH}. "
        "Asegurate de ejecutar el pipeline desde oil-gas-forecast/ o de que config.yaml exista."
    )

with _CONFIG_PATH.open() as _f:
    try:
        _cfg = yaml.safe_load(_f)
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"No se pudo parsear {_CONFIG_PATH}. "
            "Verificá la sintaxis YAML del archivo."
        ) from exc

if _cfg is None:
    raise RuntimeError(f"El archivo de configuración {_CONFIG_PATH} está vacío.")

DATA_URL = _cfg["data"]["url"]
RAW_PATH = BASE_DIR / _cfg["data"]["raw_path"]
FEAST_REPO_PATH = BASE_DIR / _cfg["feast"]["repo_path"]
ROLLING_WINDOW: int = _cfg["features"]["rolling_window"]

_FEATURES_MODULE_PATH = FEAST_REPO_PATH / "features.py"
_FEATURES_SPEC = importlib.util.spec_from_file_location("feature_store.features", _FEATURES_MODULE_PATH)
if _FEATURES_SPEC is None or _FEATURES_SPEC.loader is None:
    raise ImportError(f"No se pudo cargar el módulo de features desde {_FEATURES_MODULE_PATH}")
_FEATURES_MODULE = importlib.util.module_from_spec(_FEATURES_SPEC)
_FEATURES_SPEC.loader.exec_module(_FEATURES_MODULE)
pozo = _FEATURES_MODULE.pozo
well_stats = _FEATURES_MODULE.well_stats


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


def compute_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Computa Model-Independent Transformations (MIT).

    Estas transformaciones no dependen del algoritmo de ML final y pueden
    ser reutilizadas por múltiples modelos. Se almacenan en el Feature Store
    para garantizar consistencia entre training e inferencia.

    El tamaño de ventana rolling (`window`) se toma por defecto de config.yaml
    (clave ``features.rolling_window``) para que las columnas generadas
    reflejen siempre el valor configurado.
    """
    df = df.sort_values(["idpozo", "fecha"])
    grp = df.groupby("idpozo")

    pet_col = f"avg_prod_pet_{window}m"
    gas_col = f"avg_prod_gas_{window}m"

    # Promedio de las últimas `window` lecturas por pozo
    df[pet_col] = grp["prod_pet"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[gas_col] = grp["prod_gas"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # shift(1): última lectura conocida — evita leakage (en producción no conocemos el valor actual)
    df["last_prod_pet"] = grp["prod_pet"].shift(1).fillna(0)

    # Cantidad de registros históricos disponibles para el pozo en ese momento
    df["n_readings"] = grp.cumcount() + 1

    feature_cols = [
        "idpozo", "fecha",
        "prod_pet", "prod_gas", "prod_agua",
        pet_col, gas_col,
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

    # 3. Online store: última lectura por pozo (para inferencia en tiempo real)
    latest = df.sort_values("fecha").groupby("idpozo").tail(1).copy()
    latest["event_timestamp"] = latest["fecha"]
    store.write_to_online_store("well_stats", latest)
    logger.info(f"Online store materializado: {latest['idpozo'].nunique():,} pozos activos")


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_features = compute_features(df_clean)
    materialize_to_feast(df_features)
    logger.info("✅ Feature Pipeline completado")
