from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

pozo = Entity(
    name="idpozo",
    description="Identificador único del pozo de extracción",
)

well_stats_source = FileSource(
    path="feature_store/data/well_features.parquet",
    timestamp_field="fecha",
)

well_stats = FeatureView(
    name="well_stats",
    entities=[pozo],
    ttl=timedelta(days=365 * 5),
    schema=[
        # Producción cruda
        Field(name="prod_pet",         dtype=Float32),
        Field(name="prod_gas",         dtype=Float32),
        Field(name="prod_agua",        dtype=Float32),
        # Model-Independent Transformations (MIT)
        Field(name="avg_prod_pet_10m", dtype=Float32),
        Field(name="avg_prod_gas_10m", dtype=Float32),
        Field(name="last_prod_pet",    dtype=Float32),
        Field(name="n_readings",       dtype=Int64),
        # Características estáticas del pozo
        Field(name="profundidad",      dtype=Float32),
        Field(name="tipo_extraccion",  dtype=Int64),
    ],
    source=well_stats_source,
)
