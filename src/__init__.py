"""Source package for the traffic-volume MLOps pipeline.

The modules in this package cover data ingestion, preprocessing, model
selection, training, evaluation, prediction, registry metadata, and supporting
monitoring helpers. Keeping them importable as a package lets tests, Flask,
Docker, and GitHub Actions use the same code paths.
"""
