"""
Detector de drift y model decay para el pipeline de pronóstico de producción.

Dos métricas (cumple el mínimo requerido por la consigna):
  1. Data drift por feature — Kolmogorov-Smirnov via alibi-detect TabularDrift
  2. Model decay — comparación del MAE actual vs promedio de últimos N runs en MLflow
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftDetector:
    """Encapsula las dos métricas de monitoring sobre el feature store y MLflow."""

    # Features sobre las que se calcula drift — mismas que entran al modelo
    FEATURE_COLS = [
        "avg_prod_pet_10m",
        "avg_prod_gas_10m",
        "last_prod_pet",
        "n_readings",
        "profundidad",
        "tipo_extraccion",
    ]

    # Umbral de degradación del MAE que se considera model decay
    DECAY_THRESHOLD_PCT = 20.0

    def __init__(self, repo_path: str = "./feature_store"):
        self.repo_path = Path(repo_path)
        self.parquet_path = self.repo_path / "data" / "well_features.parquet"

    def compute_ks_drift(
        self, fecha_corte: str, window_months: int = 3
    ) -> Dict[str, Any]:
        """
        KS por feature contra un baseline fijo (primeros N meses del dataset).

        Baseline = primeros `window_months` meses del histórico (referencia estable).
        Actual = últimos `window_months` meses antes de `fecha_corte`.

        Retorna p-values por feature y un flag global `is_drift` que es True
        si al menos una feature rechaza la hipótesis nula a p < 0.05.
        """
        # Import diferido: alibi-detect puede no estar instalado en todos los entornos
        # donde se importa el módulo (ej. tests del endpoint que solo lee el JSON).
        from alibi_detect.cd import TabularDrift

        df = pd.read_parquet(self.parquet_path)
        df["fecha"] = pd.to_datetime(df["fecha"])

        fecha_corte_ts = pd.to_datetime(fecha_corte)
        fecha_inicio_actual = fecha_corte_ts - pd.DateOffset(months=window_months)

        fecha_min = df["fecha"].min()
        fecha_fin_ref = fecha_min + pd.DateOffset(months=window_months)

        ref_mask = (df["fecha"] >= fecha_min) & (df["fecha"] < fecha_fin_ref)
        curr_mask = (df["fecha"] >= fecha_inicio_actual) & (df["fecha"] < fecha_corte_ts)

        X_ref = df[ref_mask][self.FEATURE_COLS].dropna().values
        X_curr = df[curr_mask][self.FEATURE_COLS].dropna().values

        if len(X_ref) < 10 or len(X_curr) < 10:
            logger.warning(
                "Muestras insuficientes para KS drift (ref=%d, curr=%d) — skip",
                len(X_ref), len(X_curr),
            )
            return {
                "is_drift": False,
                "p_values": {},
                "threshold": 0.05,
                "skipped": True,
                "n_ref": len(X_ref),
                "n_curr": len(X_curr),
            }

        detector = TabularDrift(X_ref.astype(np.float32), p_val=0.05)
        result = detector.predict(X_curr.astype(np.float32))

        p_values = {
            col: float(result["data"]["p_val"][i])
            for i, col in enumerate(self.FEATURE_COLS)
        }

        return {
            "is_drift": bool(result["data"]["is_drift"]),
            "p_values": p_values,
            "threshold": 0.05,
            "n_ref": len(X_ref),
            "n_curr": len(X_curr),
        }

    def compute_model_decay(
        self, current_run_id: str, lookback_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Compara el val_mae del run actual contra el promedio de los últimos N runs.

        Retorna `is_decay=True` si el MAE actual supera al histórico por más de
        DECAY_THRESHOLD_PCT. Si no hay historial, retorna sin error pero con nota.
        """
        client = mlflow.tracking.MlflowClient()

        current_run = client.get_run(current_run_id)
        current_mae = current_run.data.metrics.get("val_mae")
        if current_mae is None:
            return {
                "is_decay": False,
                "error": "El run actual no tiene métrica val_mae logueada",
            }

        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "hydrocarbon-forecast")
        experiments = client.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )
        if not experiments:
            return {
                "is_decay": False,
                "error": f"Experimento '{experiment_name}' no encontrado",
            }

        runs = client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
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
            "is_decay": pct_change > self.DECAY_THRESHOLD_PCT,
            "current_mae": float(current_mae),
            "historical_mae_mean": historical_mean,
            "pct_change": float(pct_change),
            "decay_threshold_pct": self.DECAY_THRESHOLD_PCT,
            "n_historical_runs": len(historical_maes),
        }
