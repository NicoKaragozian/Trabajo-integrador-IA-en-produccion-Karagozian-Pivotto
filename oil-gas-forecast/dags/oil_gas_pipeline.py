"""
DAG mensual de produccion: feature -> training -> monitoring.

La frecuencia es @monthly porque el dataset de datos.energia.gob.ar se reporta
mes a mes. Cada run usa data_interval_end como fecha de corte para garantizar
reproducibilidad: el run del intervalo abril -> mayo siempre corta en 2026-05-01,
sin importar cuando se ejecute dentro del mes.

PYTHONPATH=/opt/airflow:/opt/airflow/pipelines:/opt/airflow/monitoring esta
seteado por el compose, asi que los imports dentro de cada @task funcionan sin
sys.path manual.
"""
import os
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "nico",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="oil_gas_pipeline",
    description="Pipeline mensual de pronostico de produccion de hidrocarburos",
    schedule_interval="@monthly",
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["oil-gas", "mlops", "forecast"],
)
def oil_gas_pipeline():

    @task(task_id="feature_pipeline")
    def feature_task(**context) -> str:
        from feature_pipeline import (
            load_raw_data, clean_data, compute_features, materialize_to_feast,
        )

        fecha_corte = context["data_interval_end"].strftime("%Y-%m-%d")

        df_raw = load_raw_data()
        df_clean = clean_data(df_raw)
        df_features = compute_features(df_clean)
        materialize_to_feast(df_features)

        return fecha_corte

    @task(task_id="training_pipeline")
    def training_task(fecha_corte: str) -> dict:
        from training_pipeline import train

        run_id, version = train(
            fecha_corte=fecha_corte,
            config_path=os.environ["CONFIG_PATH"],
        )
        return {"run_id": run_id, "version": version, "fecha_corte": fecha_corte}

    @task(task_id="monitoring_pipeline")
    def monitoring_task(training_result: dict) -> dict:
        from monitoring.report_generator import generate_report

        return generate_report(
            run_id=training_result["run_id"],
            fecha_corte=training_result["fecha_corte"],
            repo_path=os.environ["FEAST_REPO_PATH"],
            config_path=os.environ["CONFIG_PATH"],
        )

    fecha = feature_task()
    training_result = training_task(fecha)
    monitoring_task(training_result)


dag = oil_gas_pipeline()
