FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/energy_efficiency_heating_load_regressor.joblib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=120 \
    --retries 10 \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY data/raw/ data/raw/
COPY models/ models/
COPY reports/ reports/

RUN python -m compileall app src \
    && ALLOW_INSECURE_DATA_DOWNLOAD=1 python -m src.data \
    && python -m src.preprocess \
    && python -m src.train \
    && python -m src.evaluate \
    && python -m src.model_registry

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app.main:app"]
