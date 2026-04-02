# Oil & Gas Forecast — IA en Producción

Pipeline de ML para pronosticar la producción futura de pozos de hidrocarburos no convencionales (Argentina). El foco está en **ML Engineering**: Feature Store, reproducibilidad, tracking de experimentos y API REST lista para producción.

---

## Arquitectura

```
docker-compose
├── postgres          ← backend de MLflow
├── mlflow            ← tracking server + model registry  (puerto 6000)
└── api (FastAPI)     ← /forecast + /wells                (puerto 8000)
      ├── lee features del Feature Store online (Feast + SQLite)
      └── carga el modelo en Production desde MLflow Registry

Pipelines (ejecutar a mano o vía Airflow):
  feature_pipeline.py   →  datos crudos → Feature Store offline + online
  training_pipeline.py  →  Feature Store offline → XGBoost → MLflow
```

---

## Quickstart

### 1. Clonar y configurar

```bash
git clone <repo-url>
cd oil-gas-forecast
cp .env .env.local   # opcional: ajustar variables
```

### 2. Levantar la infraestructura

```bash
docker-compose up -d
```

Servicios disponibles:
- **MLflow UI:** http://localhost:6000
- **API + Swagger:** http://localhost:8000/docs

### 3. Generar el Feature Store

```bash
docker-compose exec api python pipelines/feature_pipeline.py
```

Descarga el dataset, computa las features (MIT) y materializa el Feature Store offline y online.

### 4. Entrenar el modelo

```bash
docker-compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01
```

Entrena un XGBoost con datos hasta la fecha indicada, loguea métricas y artefactos en MLflow, y registra el modelo como `Production` en el Registry.

### 5. Consultar la API

```bash
# Listar pozos activos al 1 de junio 2024
curl "http://localhost:8000/api/v1/wells?date_query=2024-06-01"

# Pronóstico de producción mensual para un pozo
curl "http://localhost:8000/api/v1/forecast?id_well=18122-001&date_start=2024-07-01&date_end=2024-12-01"
```

---

## Dataset

- **Producción de pozos (principal):** [datos.gob.ar](https://datos.gob.ar/dataset/energia-produccion-petroleo-gas-por-pozo-capitulo-iv/archivo/energia_b5b58cdc-9e07-41f9-b392-fb9ec68b0725)
- **Listado de pozos:** [datos.gob.ar](https://datos.gob.ar/dataset/energia-produccion-petroleo-gas-por-pozo-capitulo-iv/archivo/energia_cbfa4d79-ffb3-4096-bab5-eb0dde9a8385)

El `feature_pipeline.py` lo descarga automáticamente en la primera ejecución y lo cachea en `data/raw/pozos.csv`.

---

## Endpoints de la API

### `GET /api/v1/wells`

Lista los pozos con datos disponibles hasta una fecha dada.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `date_query` | `date` | Fecha de corte (`YYYY-MM-DD`) |

**Ejemplo de respuesta:**
```json
[
  {"id_well": "18122-001"},
  {"id_well": "18122-002"}
]
```

### `GET /api/v1/forecast`

Pronóstico mensual de producción de petróleo para un pozo.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `id_well` | `string` | Identificador del pozo |
| `date_start` | `date` | Inicio del horizonte (`YYYY-MM-DD`) |
| `date_end` | `date` | Fin del horizonte (`YYYY-MM-DD`) |

**Ejemplo de respuesta:**
```json
{
  "id_well": "18122-001",
  "data": [
    {"date": "2024-07-01", "prod": 142.5},
    {"date": "2024-08-01", "prod": 139.65},
    {"date": "2024-09-01", "prod": 136.86}
  ]
}
```

### `GET /health`

Estado del servicio y si el modelo está cargado.

---

## Reproducibilidad

El sistema garantiza reproducibilidad a través de tres pilares:

| Pilar | Implementación |
|-------|----------------|
| **Código** | Git + `config.yaml` con todos los hiperparámetros (nunca hardcodeados) + `random_state=42` + config logueado como artefacto en MLflow |
| **Datos** | Feast offline store con timestamp. El Point-in-Time Join con `--fecha` garantiza los mismos datos para la misma fecha de corte, siempre. |
| **Entorno** | Docker con `requirements.txt` pinneado. Cualquiera reproduce el entorno exacto con `docker-compose up`. |

**Verificación:** dos runs con la misma `--fecha` deben producir el mismo `val_mae`:

```bash
python pipelines/training_pipeline.py --fecha 2024-06-01  # Run 1
python pipelines/training_pipeline.py --fecha 2024-06-01  # Run 2 → métricas idénticas
```

---

## Estructura del proyecto

```
oil-gas-forecast/
├── docker-compose.yml       ← infraestructura completa
├── .env                     ← variables de entorno (no commitear credenciales reales)
├── config.yaml              ← hiperparámetros del modelo e inferencia
├── README.md
├── feature_store/
│   ├── feature_store.yaml   ← configuración de Feast (local, SQLite online store)
│   ├── features.py          ← Entity + FeatureView (usados en training e inferencia)
│   └── data/
│       └── well_features.parquet   ← generado por feature_pipeline.py
├── pipelines/
│   ├── feature_pipeline.py  ← datos → features → Feast (offline + online)
│   └── training_pipeline.py ← Feast → XGBoost → MLflow tracking + registry
├── api/
│   ├── main.py              ← FastAPI: /forecast, /wells, /health
│   ├── Dockerfile
│   └── requirements.txt
└── data/
    └── raw/
        └── pozos.csv        ← cacheado por feature_pipeline.py (no commitear)
```

---

## Decisiones de diseño

### Feature Store (Feast)

Sin Feature Store, las transformaciones de features estarían duplicadas: una vez en `training_pipeline.py` y otra en la API. Eso introduce **Training-Serving Skew** — el modelo no crashea, simplemente predice peor en producción y el bug es difícil de encontrar.

Con Feast, `features.py` es la única fuente de verdad. El training usa el **offline store** (con Point-in-Time Join para evitar data leakage), y la API usa el **online store** (SQLite, baja latencia). Misma lógica, cero skew.

Se eligió SQLite como online store porque el dataset tiene miles de pozos, no millones de requests por segundo. Migrar a Redis es una línea en `feature_store.yaml`, sin tocar código.

### MLflow Model Registry

La API carga siempre `models:/hydrocarbon-forecast-model/Production`. El modelo en `Production` es explícito, versionado y tiene todo el linaje (run, datos, métricas). El rollback es trivial: promover la versión anterior a `Production`.

### FastAPI

El enunciado exige documentación OpenAPI (Swagger). FastAPI la genera **automáticamente** a partir de los type hints (Pydantic). Con Flask habría que escribirla a mano. Los schemas `ForecastResponse` y `WellInfo` también validan los tipos de respuesta en tiempo de ejecución.

### Modelo: XGBoost con ventanas temporales

El foco del trabajo es ML Engineering, no la precisión del modelo. XGBoost con features de ventana temporal (promedio 10 lecturas, última lectura, n_readings) captura la estructura temporal sin la complejidad de un modelo secuencial. Es interpretable (feature importance), tiene tracking nativo en MLflow y es determinista con `random_state=42`.

### Pronóstico hacia el futuro: Decline Curve

El modelo predice la producción base de un pozo. Para generar una serie `date_start → date_end`, se aplica una curva de decline exponencial:

```
prod(mes i) = prod_base × (1 - decline_rate)^i
```

Con `decline_rate = 0.02` (configurable en `config.yaml`). Los pozos de petróleo y gas siguen patrones de decline físicamente bien estudiados. **Limitación conocida:** la tasa es global para todos los pozos; una versión más sofisticada calcularía la tasa histórica por pozo desde el Feature Store.

---

## Comandos rápidos

```bash
# Levantar todo
docker-compose up -d

# Feature Pipeline
docker-compose exec api python pipelines/feature_pipeline.py

# Training (parametrizable por fecha)
docker-compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01
docker-compose exec api python pipelines/training_pipeline.py --fecha 2023-12-01

# Endpoints
curl "http://localhost:8000/api/v1/wells?date_query=2024-06-01"
curl "http://localhost:8000/api/v1/forecast?id_well=18122-001&date_start=2024-07-01&date_end=2024-12-01"

# UIs
open http://localhost:6000       # MLflow
open http://localhost:8000/docs  # Swagger
```

---

## Equipo

- **Nico Karagozian** — Feature Pipeline, Feature Store, Training Pipeline, MLflow
- **Valentino Pivotto** — Docker Compose, infra, API REST (FastAPI), README
