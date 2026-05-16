"""
Generador del reporte consolidado de monitoring.

Junta data drift + model decay en un único JSON que:
  - Se persiste en disco para que el endpoint `/monitoring/report` lo sirva.
  - Se loguea en MLflow como artefacto del run, junto con métricas escalares
    (drift_detected, decay_detected, mae_pct_change) para que queden visibles
    en la UI de tracking.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)

# Path del JSON persistido. Default pensado para correr local desde el repo;
# en docker se sobreescribe vía env var para apuntar al volumen compartido.
DEFAULT_REPORT_PATH = "./data/monitoring_report.json"


def generate_report(
    run_id: str,
    fecha_corte: str,
    repo_path: str = "./feature_store",
) -> dict:
    """
    Ejecuta las dos métricas, arma el reporte, lo persiste y lo loguea en MLflow.

    Devuelve el reporte como dict por si el caller lo quiere inspeccionar
    (ej. el monitoring_task del DAG, para pasarlo por XCom).
    """
    # Setear la URI antes de tocar MLflow — compute_model_decay() también
    # instancia MlflowClient internamente, así que la URI tiene que estar
    # configurada antes de invocarlo (sino usa el default ./mlruns).
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6000"))

    detector = DriftDetector(repo_path=repo_path)

    drift_result = detector.compute_ks_drift(fecha_corte, window_months=3)
    decay_result = detector.compute_model_decay(run_id, lookback_runs=5)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fecha_corte": fecha_corte,
        "run_id": run_id,
        "data_drift": drift_result,
        "model_decay": decay_result,
        "alerts": [],
    }

    if drift_result.get("is_drift"):
        threshold = drift_result.get("threshold", 0.05)
        drifted = [
            f for f, p in drift_result.get("p_values", {}).items() if p < threshold
        ]
        report["alerts"].append({
            "type": "data_drift",
            "level": "warning",
            "message": f"Drift detectado en features: {', '.join(drifted)}",
        })

    if decay_result.get("is_decay"):
        report["alerts"].append({
            "type": "model_decay",
            "level": "critical",
            "message": (
                f"Model decay: MAE actual={decay_result['current_mae']:.2f} "
                f"vs histórico={decay_result['historical_mae_mean']:.2f} "
                f"({decay_result['pct_change']:+.1f}%)"
            ),
        })

    report_path = Path(os.getenv("MONITORING_REPORT_PATH", DEFAULT_REPORT_PATH))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Reporte de monitoring escrito en %s", report_path)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(report, "monitoring/report.json")
        mlflow.log_metric("drift_detected", int(bool(drift_result.get("is_drift"))))
        mlflow.log_metric("decay_detected", int(bool(decay_result.get("is_decay"))))
        pct = decay_result.get("pct_change")
        if pct is not None:
            mlflow.log_metric("mae_pct_change", float(pct))

    return report
