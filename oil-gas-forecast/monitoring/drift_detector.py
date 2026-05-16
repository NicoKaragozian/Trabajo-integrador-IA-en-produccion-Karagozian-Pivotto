"""
Detector de drift y model decay para el pipeline de pronóstico de producción.

Dos métricas (cumple el mínimo requerido por la consigna):
  1. Data drift por feature — Kolmogorov-Smirnov via alibi-detect TabularDrift
  2. Model decay — comparación del MAE actual vs promedio de últimos N runs en MLflow
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftDetector:
    """Encapsula las dos métricas de monitoring sobre el feature store y MLflow.

    Todos los hiperparámetros (umbrales, ventanas, lookback) se reciben en
    `config`, que es la sección `monitoring` de config.yaml. No hay defaults
    en el código — la fuente única de verdad es el archivo de configuración.

    `feature_cols` se pasa explícitamente (typicalmente desde `model.features`
    del config) para garantizar que el drift se mide sobre las mismas features
    que usa el modelo.
    """

    def __init__(
        self,
        repo_path: str,
        config: Dict[str, Any],
        feature_cols: List[str],
    ):
        self.repo_path = Path(repo_path)
        self.parquet_path = self.repo_path / "data" / "well_features.parquet"
        self.config = config
        self.feature_cols = feature_cols

    def compute_ks_drift(self, fecha_corte: str) -> Dict[str, Any]:
        """
        KS por feature contra un baseline fijo (primeros N meses del dataset).

        Baseline = primeros `window_months` meses del histórico (referencia estable).
        Actual = últimos `window_months` meses antes de `fecha_corte`.

        Retorna p-values por feature y un flag global `is_drift` que es True
        si al menos una feature rechaza la hipótesis nula a p < p_value_threshold.
        """
        # Import diferido: alibi-detect puede no estar instalado en todos los entornos
        # donde se importa el módulo (ej. tests del endpoint que solo lee el JSON).
        from alibi_detect.cd import TabularDrift

        window_months = self.config["drift"]["window_months"]
        p_val_threshold = self.config["drift"]["p_value_threshold"]
        min_samples = self.config["drift"]["min_samples"]

        df = pd.read_parquet(self.parquet_path)
        df["fecha"] = pd.to_datetime(df["fecha"])

        fecha_corte_ts = pd.to_datetime(fecha_corte)
        fecha_inicio_actual = fecha_corte_ts - pd.DateOffset(months=window_months)

        fecha_min = df["fecha"].min()
        fecha_fin_ref = fecha_min + pd.DateOffset(months=window_months)

        ref_mask = (df["fecha"] >= fecha_min) & (df["fecha"] < fecha_fin_ref)
        curr_mask = (df["fecha"] >= fecha_inicio_actual) & (df["fecha"] < fecha_corte_ts)

        X_ref = df[ref_mask][self.feature_cols].dropna().values
        X_curr = df[curr_mask][self.feature_cols].dropna().values

        if len(X_ref) < min_samples or len(X_curr) < min_samples:
            logger.warning(
                "Muestras insuficientes para KS drift (ref=%d, curr=%d, min=%d) — skip",
                len(X_ref), len(X_curr), min_samples,
            )
            return {
                "is_drift": False,
                "p_values": {},
                "threshold": p_val_threshold,
                "skipped": True,
                "n_ref": len(X_ref),
                "n_curr": len(X_curr),
            }

        detector = TabularDrift(X_ref.astype(np.float32), p_val=p_val_threshold)
        result = detector.predict(X_curr.astype(np.float32))

        p_values = {
            col: float(result["data"]["p_val"][i])
            for i, col in enumerate(self.feature_cols)
        }

        return {
            "is_drift": bool(result["data"]["is_drift"]),
            "p_values": p_values,
            "threshold": p_val_threshold,
            "n_ref": len(X_ref),
            "n_curr": len(X_curr),
        }

    def compute_model_decay(self, current_run_id: str) -> Dict[str, Any]:
        """
        Compara el val_mae del run actual contra el promedio de los últimos N runs.

        Retorna `is_decay=True` si el MAE actual supera al histórico por más de
        `decay.threshold_pct`. Si no hay historial, retorna sin error pero con nota.
        """
        lookback_runs = self.config["decay"]["lookback_runs"]
        decay_threshold_pct = self.config["decay"]["threshold_pct"]

        client = mlflow.tracking.MlflowClient()

        current_run = client.get_run(current_run_id)
        current_mae = current_run.data.metrics.get("val_mae")
        if current_mae is None:
            return {
                "is_decay": False,
                "error": "El run actual no tiene métrica val_mae logueada",
            }

        # Usar el experimento del propio run para que la comparación sea consistente
        # independientemente de qué env var de MLFLOW_EXPERIMENT_NAME esté seteada.
        runs = client.search_runs(
            experiment_ids=[current_run.info.experiment_id],
            filter_string="metrics.val_mae > 0",
            order_by=["start_time DESC"],
            max_results=lookback_runs + 1,  # +1 para excluir el run actual del lookback
        )

        historical_maes = [
            r.data.metrics["val_mae"]
            for r in runs
            if r.info.run_id != current_run_id
        ][:lookback_runs]

        if not historical_maes:
            return {
                "is_decay": False,
                "current_mae": float(current_mae),
                "historical_mae_mean": None,
                "pct_change": 0.0,
                "note": "Sin runs históricos para comparar",
            }

        historical_mean = float(np.mean(historical_maes))
        pct_change = (current_mae - historical_mean) / historical_mean * 100

        return {
            "is_decay": pct_change > decay_threshold_pct,
            "current_mae": float(current_mae),
            "historical_mae_mean": historical_mean,
            "pct_change": float(pct_change),
            "decay_threshold_pct": decay_threshold_pct,
            "n_historical_runs": len(historical_maes),
        }
