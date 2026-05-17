"""Entrypoint de Ray Serve: envuelve la app FastAPI de main.py en un Deployment."""
import argparse
import logging
import os
import time

import ray
from ray import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    ray.init(ignore_reinit_error=True)
    serve.run(APIIngress.bind(), host="0.0.0.0", port=8000)
    logger.info("Ray Serve up con %d replicas en :8000", num_replicas)

    # mantener el proceso vivo (serve corre en actors background)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replicas",
        type=int,
        default=int(os.getenv("RAY_NUM_REPLICAS", "2")),
    )
    args = parser.parse_args()
    run(args.replicas)
