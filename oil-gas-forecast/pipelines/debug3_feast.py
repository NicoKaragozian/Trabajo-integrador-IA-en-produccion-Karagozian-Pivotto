"""Debug definitivo: qué event_ts está guardado y por qué Feast retorna None."""
import sqlite3
import pandas as pd
from feast import FeatureStore
from feast.infra.key_encoding_utils import serialize_entity_key
from feast.protos.feast.types.EntityKey_pb2 import EntityKey
from feast.protos.feast.types.Value_pb2 import Value
import datetime

store = FeatureStore(repo_path="feature_store")
fv = store.get_feature_view("well_stats")
print(f"TTL: {fv.ttl}")

# Build v2 key for idpozo=3640
ek = EntityKey()
ek.join_keys.append("idpozo")
v = Value()
v.int64_val = 3640
ek.entity_values.append(v)
key_v2 = serialize_entity_key(ek, entity_key_serialization_version=2)

conn = sqlite3.connect("feature_store/online_store.db")
cursor = conn.cursor()

# Show raw event_ts for idpozo=3640
cursor.execute(
    "SELECT feature_name, event_ts, typeof(event_ts), value FROM oil_gas_forecast_well_stats WHERE entity_key = ? LIMIT 10",
    (key_v2,)
)
rows = cursor.fetchall()
print(f"\nRows for idpozo=3640 (v2 key):")
for fn, ev_ts, ts_type, val in rows:
    print(f"  feature={fn}, event_ts={ev_ts!r}, type={ts_type}")

conn.close()

# Now do get_online_features and inspect full response
print("\n=== Full get_online_features response ===")
result = store.get_online_features(
    features=["well_stats:avg_prod_pet_10m", "well_stats:last_prod_pet"],
    entity_rows=[{"idpozo": 3640}],
).to_df()
print(result)
print("\nDtypes:")
print(result.dtypes)
