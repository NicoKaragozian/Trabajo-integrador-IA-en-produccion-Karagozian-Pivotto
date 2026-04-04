"""Simula exactamente lo que hace la API en el endpoint /forecast."""
import os
from feast import FeatureStore

FEAST_REPO = os.getenv("FEAST_REPO_PATH", "/app/feature_store")
MODEL_NAME = os.getenv("MODEL_NAME", "hydrocarbon-forecast-model")

FEAST_FEATURES = [
    "well_stats:avg_prod_pet_10m",
    "well_stats:avg_prod_gas_10m",
    "well_stats:last_prod_pet",
    "well_stats:n_readings",
    "well_stats:profundidad",
    "well_stats:tipo_extraccion",
]

print(f"FEAST_REPO: {FEAST_REPO}")
STORE = FeatureStore(repo_path=FEAST_REPO)

id_well = "3640"
entity_id = int(id_well) if id_well.isdigit() else id_well
print(f"entity_id: {entity_id} (type: {type(entity_id).__name__})")

online = STORE.get_online_features(
    features=FEAST_FEATURES,
    entity_rows=[{"idpozo": entity_id}],
).to_dict()

print("\nResult keys:", list(online.keys()))
for k, v in online.items():
    print(f"  {k}: {v}")

# Check what API checks
check_val = online.get("well_stats__avg_prod_pet_10m", [None])[0]
print(f"\nCheck value (avg_prod_pet_10m[0]): {check_val!r}")
print(f"Is None: {check_val is None}")
if check_val is None:
    print("→ API would return 404")
else:
    print("→ API would return forecast data")
