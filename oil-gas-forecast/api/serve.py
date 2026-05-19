"""Entrypoint de Ray Serve: envuelve la app FastAPI de main.py en un Deployment."""
import argparse
import logging
import os
import signal
import threading

import ray
from ray import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_shutdown = threading.Event()


def _parse_replicas(value: str | None) -> int:
    # tolera RAY_NUM_REPLICAS vacio en .env (int("") rompe)
    raw = (value or "").strip() or "2"
    n = int(raw)
    if n < 1:
        raise ValueError(f"RAY_NUM_REPLICAS debe ser >= 1, recibido {n}")
    return n


def _handle_signal(signum, _frame):
    logger.info("recibida senal %s, iniciando shutdown", signum)
    _shutdown.set()


def run(num_replicas: int) -> None:
    # importar la app dentro de la funcion para que el driver no cargue el modelo
    from main import app

    @serve.deployment(
        name="oil-gas-forecast-api",
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": 1},
    )
    @serve.ingress(app)
    class APIIngress:
        pass

    # include_dashboard=False evita levantar el dashboard de ray dentro del container
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    serve.run(APIIngress.bind(), host="0.0.0.0", port=8000)
    logger.info("Ray Serve up con %d replicas en :8000", num_replicas)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _shutdown.wait()

    logger.info("apagando serve + ray")
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replicas",
        type=int,
        default=_parse_replicas(os.getenv("RAY_NUM_REPLICAS")),
    )
    args = parser.parse_args()
    run(args.replicas)
