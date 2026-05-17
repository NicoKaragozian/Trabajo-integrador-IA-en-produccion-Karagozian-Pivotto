FROM apache/airflow:2.8.1-python3.10

COPY pipelines/requirements.txt /tmp/pipelines-requirements.txt
RUN pip install --no-cache-dir -r /tmp/pipelines-requirements.txt
