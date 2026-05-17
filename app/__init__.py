"""Flask application package for serving the trained wine-quality model.

`app.main` creates the API and browser UI, `app.schemas` owns the prediction
contract, and `app.model_loader` loads the saved joblib bundle. Tests and Docker
import this package so the serving path is checked before deployment.
"""

from app.main import create_app

__all__ = ["create_app"]
