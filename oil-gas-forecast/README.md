# Oil & Gas Forecast

Plataforma de pronóstico de producción de hidrocarburos no convencionales.
Trabajo Integrador — IA en Producción · MIA204 · UdeSA 2026

**Equipo:** Nicolás Karagozian · Valentina Pivotto

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

---

## Cómo correr el feature pipeline

```bash
cd oil-gas-forecast
pip install feast pandas pyarrow

python pipelines/feature_pipeline.py
```

Genera:
- `feature_store/data/well_features.parquet` — offline store con historial completo
- `feature_store/registry.db` — definiciones de Feast (Entity + FeatureView)
- `feature_store/online_store.db` — última lectura por pozo para inferencia

---

## Estructura del proyecto

```
oil-gas-forecast/
├── feature_store/
│   ├── feature_store.yaml     ← config de Feast
│   ├── features.py            ← Entity (idpozo) + FeatureView (well_stats)
│   └── data/                  ← offline store (generado, no commiteado)
├── pipelines/
│   ├── feature_pipeline.py    ← datos crudos → features → Feast
│   ├── training_pipeline.py   ← Feast → entrena XGBoost → MLflow (próximo)
│   └── inference_pipeline.py  ← Feast online → predicción (próximo)
├── api/
│   └── main.py                ← FastAPI /api/v1/forecast y /api/v1/wells (próximo)
├── notebooks/
│   └── exploracion.ipynb      ← EDA del dataset
├── data/
│   └── raw/                   ← datos crudos (no commiteados)
└── mlruns/                    ← artefactos de MLflow (no commiteados)
```

---

## Decisiones de diseño

### Feature Store con Feast (provider local)

Usamos Feast con `provider: local` para poder correr el sistema sin infraestructura externa. El offline store es un archivo Parquet y el online store es SQLite — suficiente para el scope del trabajo.

**Por qué un Feature Store y no procesar directo al modelo:**
- Garantiza que training e inferencia usan exactamente las mismas features (elimina *training-serving skew*)
- Las features son reutilizables por múltiples modelos sin reprocesar
- Permite Point-in-Time joins: al entrenar con `fecha_corte`, solo usamos datos que hubieran estado disponibles en esa fecha

### Features MIT (Model-Independent Transformations)

Las features que almacenamos en Feast son transformaciones que **no dependen del algoritmo final**:

| Feature | Descripción | Decisión |
|---|---|---|
| `avg_prod_pet_10m` | Promedio rolling de 10 meses de producción de petróleo | Ventana de 10 meses: balance entre capturar tendencia y no requerir historia muy larga |
| `avg_prod_gas_10m` | Idem para gas | |
| `last_prod_pet` | Producción del mes anterior (`shift(1)`) | `shift(1)` evita data leakage: en producción no conocemos el valor del mes actual |
| `n_readings` | Cantidad de registros históricos disponibles | Feature de confianza: el modelo puede ponderar distinto pozos con poco historial |
| `profundidad` | Profundidad del pozo (m) | Característica estática, correlaciona con tipo de formación |
| `tipo_extraccion` | Tipo de extracción codificado como entero | Encodeo ordinal simple; suficiente para XGBoost (no necesita OHE) |

### Limpieza de datos

- **Negativos en producción:** El dataset tiene valores como -0.001 m³ y -12.26 Mm³ (errores de medición). Se clampean a 0.
- **`tipoextraccion` nulos (601 casos, 0.15%):** Se asigna categoría 0 = "Sin sistema", que es semánticamente correcto para pozos en espera o inactivos.
- **`profundidad` outliers:** El máximo es 378.939 m (imposible). Se reemplaza con la media del pozo; si el pozo no tiene ningún dato válido, se usa 0.
- **`min_periods=1` en rolling:** Los pozos nuevos con pocos registros no quedan excluidos — usan el promedio de los registros disponibles.

### Target del modelo

`prod_pet` (producción de petróleo en m³/mes). Distribución muy sesgada a la derecha con ~50% de valores cero (pozos inactivos). El modelo debe manejar esto; se verá en el training pipeline.
