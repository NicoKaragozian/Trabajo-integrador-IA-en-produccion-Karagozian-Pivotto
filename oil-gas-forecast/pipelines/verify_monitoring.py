"""
Script de verificación del módulo de monitoring.

Verifica que:
1. El DriftDetector puede leer el offline store y calcular KS drift
2. El DriftDetector puede leer los runs de MLflow y calcular model decay
3. generate_report() produce el JSON consolidado, lo persiste y loguea en MLflow

Uso:
    cd oil-gas-forecast
    python pipelines/verify_monitoring.py --fecha 2024-06-01
    python pipelines/verify_monitoring.py --fecha 2024-06-01 --run-id <run_id>

Si no se pasa --run-id, el script toma el último run del experimento por defecto.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift_detector import DriftDetector
from monitoring.report_generator import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_latest_run_id(experiment_name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'")
    assert experiments, f"Experimento '{experiment_name}' no encontrado en MLflow"

    runs = client.search_runs(
        experiment_ids=[experiments[0].experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert runs, f"No hay runs en el experimento '{experiment_name}'"
    return runs[0].info.run_id


def verify_ks_drift(detector: DriftDetector, fecha_corte: str):
    result = detector.compute_ks_drift(fecha_corte, window_months=3)
    assert "is_drift" in result, "Falta is_drift en el resultado de KS drift"
    assert "p_values" in result, "Falta p_values en el resultado de KS drift"

    if result.get("skipped"):
        logger.warning(
            "⚠️  KS drift skipped por muestras insuficientes (ref=%d, curr=%d)",
            result["n_ref"], result["n_curr"],
        )
    else:
        logger.info(
            "✅ KS drift: is_drift=%s — features evaluadas=%d, n_ref=%d, n_curr=%d",
            result["is_drift"], len(result["p_values"]),
            result["n_ref"], result["n_curr"],
        )
        for feature, p in result["p_values"].items():
            marker = "⚠️ " if p < result["threshold"] else "   "
            logger.info("   %s%-22s p=%.4f", marker, feature, p)


def verify_model_decay(detector: DriftDetector, run_id: str):
    result = detector.compute_model_decay(run_id, lookback_runs=5)
    assert "is_decay" in result, "Falta is_decay en el resultado de model decay"

    if result.get("error"):
        logger.warning("⚠️  Model decay: %s", result["error"])
        return

    if result.get("historical_mae_mean") is None:
        logger.info("ℹ️  Sin historial para comparar — current_mae=%.2f", result["current_mae"])
        return

    logger.info(
        "✅ Model decay: is_decay=%s — MAE actual=%.2f vs histórico=%.2f (%.1f%%)",
        result["is_decay"],
        result["current_mae"],
        result["historical_mae_mean"],
        result["pct_change"],
    )


def verify_report_generation(run_id: str, fecha_corte: str, repo_path: str):
    report = generate_report(run_id, fecha_corte, repo_path)

    report_path = Path(os.getenv("MONITORING_REPORT_PATH", "./data/monitoring_report.json"))
    assert report_path.exists(), f"El reporte no se persistió en {report_path}"

    required_keys = ["generated_at", "fecha_corte", "run_id", "data_drift", "model_decay", "alerts"]
    for k in required_keys:
        assert k in report, f"Falta key '{k}' en el reporte"

    logger.info("✅ Reporte generado y persistido en %s", report_path)
    logger.info("   Alertas activas: %d", len(report["alerts"]))
    for alert in report["alerts"]:
        logger.info("   [%s] %s", alert["level"].upper(), alert["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar el módulo de monitoring")
    parser.add_argument("--fecha", default="2024-06-01", help="Fecha de corte YYYY-MM-DD")
    parser.add_argument("--run-id", default=None, help="Run de MLflow (default: último del experimento)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6000"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "hydrocarbon-forecast")
    repo_path = os.getenv("FEAST_REPO_PATH", "./feature_store")

    run_id = args.run_id or get_latest_run_id(experiment_name)
    logger.info("Usando run_id=%s", run_id)

    detector = DriftDetector(repo_path=repo_path)

    print("\n=== Verificación del módulo de monitoring ===\n")
    verify_ks_drift(detector, args.fecha)
    verify_model_decay(detector, run_id)
    verify_report_generation(run_id, args.fecha, repo_path)
    print("\n✅ Todas las verificaciones pasaron correctamente.\n")
