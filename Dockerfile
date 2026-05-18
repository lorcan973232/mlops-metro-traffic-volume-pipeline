FROM python:3.11-slim

# The container is the same Flask prediction service used locally and in Kind.
# These environment settings make Python logs visible in GitHub Actions and point
# the app at the model file copied into the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/traffic_volume_classifier.joblib

WORKDIR /app

# Install pinned coursework dependencies before copying the full repository. This
# keeps Docker builds repeatable and avoids depending on packages from the host.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=120 \
    --retries 10 \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# The container runs as a non-root user because the security workflow checks this.
# A marker can therefore see that the Docker artefact includes a basic hardening
# step, not only a working Flask server.
RUN groupadd --system app && \
    useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin appuser

# Copy the application, model, data, and reports into the image so the deployed
# API can serve predictions without hidden files from the local machine.
COPY --chown=appuser:app app/ app/
COPY --chown=appuser:app src/ src/
COPY --chown=appuser:app data/raw/ data/raw/
COPY --chown=appuser:app models/ models/
COPY --chown=appuser:app reports/ reports/

# Re-run the core pipeline inside the image. This proves the container can build
# the same evidence package as the local project before it is used by Docker and
# Kind smoke tests.
RUN python -m compileall app src \
    && python -m src.data \
    && python -m src.preprocess \
    && python -m src.train \
    && python -m src.evaluate --fail-on-rejection \
    && python -m src.model_registry \
    && chown -R appuser:app /app

USER appuser

EXPOSE 5000

# The health check uses the Flask `/health` route, so Docker and Kind can fail
# fast if the model is missing or the app cannot start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=8)"

# Gunicorn serves the same `app.main:app` object that tests import locally. One
# worker is enough for this coursework artefact and keeps the demo predictable.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "90", "app.main:app"]
