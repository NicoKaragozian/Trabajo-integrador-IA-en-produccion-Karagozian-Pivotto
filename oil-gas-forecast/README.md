# Oil & Gas Forecast — IA en Producción

Plataforma de pronóstico de producción de hidrocarburos no convencionales.
Trabajo Integrador — IA en Producción · MIA204 · UdeSA 2026

**Equipo:** Nicolás Karagozian · Valentino Pivotto

---

## Descripción

Sistema de ML Engineering para pronosticar la producción futura de pozos de gas y petróleo no convencional. El foco está en los procesos de ingeniería (Feature Store, tracking de experimentos, API REST), no en la sofisticación del modelo predictivo.

**Dataset:** Producción de Pozos No Convencionales — Secretaría de Energía, Argentina (datos.gob.ar)
~396.000 registros mensuales · 4.833 pozos · 2006–2026

---

## Arquitectura

```
datos.gob.ar
     │
     ▼
feature_pipeline.py  ──────────────────────────────────┐
     │                                                  │
     ▼                                                  ▼
Feature Store (Feast)                           offline store
  offline: Parquet                              (Parquet histórico)
  online:  SQLite                                       │
     │                                                  ▼
     │                                        training_pipeline.py
     │                                                  │
     │◄─────────────────── Point-in-Time Join ──────────┘
     │                                                  │
     │                                                  ▼
     │                                            MLflow Registry
     │                                           (modelo Production)
     ▼                                                  │
online store  ──────────────────────────────► FastAPI /forecast
```

**Servicios docker-compose:**
```
├── postgres   ← backend de MLflow
├── mlflow     ← tracking server + model registry  (puerto 6000)
└── api        ← FastAPI /forecast + /wells        (puerto 8000)
```

---

## Quickstart

### 1. Clonar y configurar

```bash
git clone https://github.com/NicoKaragozian/Trabajo-integrador-IA-en-produccion-Karagozian-Pivotto.git
cd Trabajo-integrador-IA-en-produccion-Karagozian-Pivotto/oil-gas-forecast
cp .env.example .env   # ajustar si es necesario
```

### 2. Levantar la infraestructura

```bash
docker compose up -d --build
```

Servicios disponibles:
- **MLflow UI:** http://localhost:6000
- **API + Swagger:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### 3. Generar el Feature Store

```bash
docker compose exec api python pipelines/feature_pipeline.py
```

Descarga el dataset (~396k registros), computa features MIT y materializa el Feature Store offline (Parquet) y online (SQLite).

### 4. Verificar el Feature Store (opcional)

```bash
docker compose exec api python pipelines/verify_feast.py
```

Simula consultas al online store (como haría la API) y un Point-in-Time join (como haría el training pipeline).

### 5. Entrenar el modelo

```bash
docker compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01
```

Entrena XGBoost con datos hasta la fecha indicada, loguea métricas y artefactos en MLflow, registra el modelo en Production.

### 6. Consultar la API

```bash
# Listar pozos activos al 1 de junio 2024
curl "http://localhost:8000/api/v1/wells?date_query=2024-06-01"

# Pronóstico mensual para un pozo (julio–diciembre 2024)
curl "http://localhost:8000/api/v1/forecast?id_well=132879&date_start=2024-07-01&date_end=2024-12-01"
```

---

## Endpoints de la API

### `GET /api/v1/wells`

Lista los pozos con datos disponibles hasta una fecha dada.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `date_query` | `date` | Fecha de corte (`YYYY-MM-DD`) |

**Respuesta:**
```json
[{"id_well": "132879"}, {"id_well": "132491"}]
```

### `GET /api/v1/forecast`

Pronóstico mensual de producción de petróleo para un pozo.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `id_well` | `string` | Identificador del pozo |
| `date_start` | `date` | Inicio del horizonte (`YYYY-MM-DD`) |
| `date_end` | `date` | Fin del horizonte (`YYYY-MM-DD`) |

**Respuesta:**
```json
{
  "id_well": "132879",
  "data": [
    {"date": "2024-07-01", "prod": 142.5},
    {"date": "2024-08-01", "prod": 139.65}
  ]
}
```

### `GET /health`

Estado del servicio. `model_loaded: false` antes de correr el training pipeline.

---

## Reproducibilidad

| Pilar | Implementación |
|-------|----------------|
| **Código** | Git + `config.yaml` con todos los hiperparámetros (nunca hardcodeados) + `random_state=42` + config logueado como artefacto en MLflow |
| **Datos** | Feast offline store con timestamp. El Point-in-Time Join con `--fecha` garantiza los mismos datos para la misma fecha de corte, siempre. |
| **Entorno** | Docker con `requirements.txt` pinneado. `docker compose up` reproduce el entorno exacto en cualquier máquina. |

**Verificación:** dos runs con la misma `--fecha` deben producir el mismo `val_mae`:

```bash
docker compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01  # Run 1
docker compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01  # Run 2 → métricas idénticas
```

---

## Estructura del proyecto

```
oil-gas-forecast/
├── docker-compose.yml       ← infraestructura completa
├── .env.example             ← template de variables de entorno
├── config.yaml              ← hiperparámetros del modelo e inferencia
├── mlflow/
│   └── Dockerfile           ← imagen custom de MLflow
├── feature_store/
│   ├── feature_store.yaml   ← config de Feast (local, SQLite online store)
│   ├── features.py          ← Entity (idpozo) + FeatureView (well_stats)
│   └── data/                ← offline store (generado, no commiteado)
├── pipelines/
│   ├── feature_pipeline.py  ← datos crudos → features → Feast offline + online
│   ├── training_pipeline.py ← Feast offline → XGBoost → MLflow tracking + registry
│   └── verify_feast.py      ← verifica que el Feature Store funciona correctamente
├── api/
│   ├── main.py              ← FastAPI: /forecast, /wells, /health
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   └── exploracion.ipynb    ← EDA del dataset
└── data/
    └── raw/                 ← datos crudos (no commiteados)
```

---

## Decisiones de diseño

### Feature Store con Feast (provider local)

Sin Feature Store, las transformaciones de features estarían duplicadas en `training_pipeline.py` y en la API, introduciendo **Training-Serving Skew**. Con Feast, `features.py` es la única fuente de verdad: el training usa el **offline store** con Point-in-Time Join (sin data leakage), y la API usa el **online store** (SQLite, baja latencia).

Se eligió SQLite para el online store porque el dataset tiene miles de pozos, no millones de requests por segundo. Migrar a Redis es una línea en `feature_store.yaml`.

### Features MIT (Model-Independent Transformations)

| Feature | Descripción |
|---------|-------------|
| `avg_prod_pet_10m` | Promedio rolling de 10 meses — captura tendencia sin requerir historia muy larga |
| `avg_prod_gas_10m` | Idem para gas |
| `last_prod_pet` | `shift(1)` — evita data leakage: en producción no conocemos el valor actual |
| `n_readings` | Historial disponible del pozo — permite al modelo ponderar pozos con poca historia |
| `profundidad` | Característica estática — correlaciona con tipo de formación |
| `tipo_extraccion` | Encodeo ordinal — suficiente para XGBoost |

### MLflow Model Registry

La API carga siempre `models:/hydrocarbon-forecast-model/Production`. El modelo en Production es explícito, versionado y tiene todo el linaje (run, datos, métricas). El rollback es trivial: promover la versión anterior a Production.

### FastAPI

El enunciado exige documentación OpenAPI (Swagger). FastAPI la genera **automáticamente** a partir de los type hints (Pydantic). Los schemas `ForecastResponse` y `WellInfo` validan los tipos de respuesta en tiempo de ejecución.

### Modelo: XGBoost con ventanas temporales

XGBoost + features de ventana temporal captura la estructura temporal sin la complejidad de un modelo secuencial. Es interpretable (feature importance), tiene tracking nativo en MLflow y es determinista con `random_state=42`.

### Pronóstico hacia el futuro: Decline Curve

Para generar la serie `date_start → date_end`, se aplica decline exponencial sobre la predicción base:

```
prod(mes i) = prod_base × (1 - decline_rate)^i
```

Con `decline_rate = 0.02` (configurable en `config.yaml`). **Limitación conocida:** la tasa es global para todos los pozos; una versión más sofisticada la calcularía por pozo desde el Feature Store.

---

## Comandos rápidos

```bash
docker compose up -d --build
docker compose exec api python pipelines/feature_pipeline.py
docker compose exec api python pipelines/verify_feast.py
docker compose exec api python pipelines/training_pipeline.py --fecha 2024-06-01
curl "http://localhost:8000/api/v1/wells?date_query=2024-06-01"
curl "http://localhost:8000/api/v1/forecast?id_well=132879&date_start=2024-07-01&date_end=2024-12-01"
open http://localhost:6000       # MLflow
open http://localhost:8000/docs  # Swagger
```
