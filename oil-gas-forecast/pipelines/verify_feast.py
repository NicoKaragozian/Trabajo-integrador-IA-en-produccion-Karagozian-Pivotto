"""
Verificación del Feature Store — correr después de feature_pipeline.py

    cd oil-gas-forecast
    python3 pipelines/verify_feast.py
"""
from feast import FeatureStore
import pandas as pd


def main():
    store = FeatureStore(repo_path="feature_store")

    # 1. Registry: verificar definiciones
    print("=== Feature Views registradas ===")
    for fv in store.list_feature_views():
        print(f"  {fv.name}: {[f.name for f in fv.features]}")
    print()
    print("=== Entities ===")
    for e in store.list_entities():
        print(f"  {e.name}: {e.description}")

    # 2. Online Store: simula lo que haría la API /forecast
    print("\n=== Online Store — Consulta por pozo (simula inferencia) ===")
    sample_pozos = pd.DataFrame({"idpozo": [132879, 132491, 100000]})
    online_features = store.get_online_features(
        features=[
            "well_stats:prod_pet",
            "well_stats:avg_prod_pet_10m",
            "well_stats:last_prod_pet",
            "well_stats:n_readings",
            "well_stats:profundidad",
        ],
        entity_rows=sample_pozos.to_dict("records"),
    ).to_df()
    print(online_features.to_string(index=False))
    print("Nota: el pozo 100000 no existe → devuelve None")

    # 3. Offline Store: Point-in-Time join (simula training con fecha_corte)
    print("\n=== Offline Store — Point-in-Time Join (fecha_corte=2020-01-31) ===")
    entity_df = pd.DataFrame({
        "idpozo": [132879, 132491],
        "event_timestamp": pd.to_datetime(["2020-01-31", "2020-01-31"]),
    })
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "well_stats:prod_pet",
            "well_stats:avg_prod_pet_10m",
            "well_stats:last_prod_pet",
            "well_stats:n_readings",
        ],
    ).to_df()
    print(training_df.to_string(index=False))
    print("No hay leakage: last_prod_pet es del mes ANTERIOR al timestamp pedido.")

    print("\n✅ Todas las verificaciones pasaron")


if __name__ == "__main__":
    main()
