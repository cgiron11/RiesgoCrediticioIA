"""Punto de entrada de la API FastAPI.

Uso:  uvicorn api.main:app --host 0.0.0.0 --port 8000

Arquitectura en capas:
  routes/      -> definicion de endpoints (HTTP puro)
  controllers/ -> mapeo de errores a codigos HTTP
  services/    -> logica de inferencia
  models/      -> acceso a los artefactos de ML
  schemas/     -> contratos Pydantic
"""

from fastapi import FastAPI

from api.models.artifacts import ARTIFACTS
from api.routes import health_router, prediction_router




def create_app() -> FastAPI:
    """Fabrica la aplicacion, registra las rutas y carga los artefactos."""
    app = FastAPI(
        title="API de Riesgo Crediticio",
        version="1.0.0",
        description=(
            "Predice el nivel de riesgo crediticio (0=bajo, 1=medio, 2=alto) a partir "
            "de las 10 caracteristicas usadas en el entrenamiento."
        ),
    )
    app.include_router(health_router)
    app.include_router(prediction_router)
    ARTIFACTS.load()

    
    return app


app = create_app()
