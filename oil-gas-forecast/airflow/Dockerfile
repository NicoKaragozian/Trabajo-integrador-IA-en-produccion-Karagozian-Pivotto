FROM apache/airflow:2.8.1-python3.10

COPY pipelines/requirements.txt /tmp/pipelines-requirements.txt
# email-validator lo trae airflow base pero la reinstalacion de fastapi/pydantic
# (al instalar feast) lo desplaza; lo fijamos para no romper imports
RUN pip install --no-cache-dir -r /tmp/pipelines-requirements.txt "email-validator>=2.0"
