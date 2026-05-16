-- corre solo cuando el volumen postgres-data esta vacio (primer up desde cero)
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO mlflow;
