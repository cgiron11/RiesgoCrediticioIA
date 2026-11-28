"""Ruta GET /health."""

from fastapi import APIRouter

from api.controllers import PredictionController
from api.schemas import HealthResponse

router = APIRouter(tags=["salud"])
controller = PredictionController()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Comprueba que el modelo esta cargado y operativo."""
    return controller.health()

 
@router.get("/", tags=["Root"])
def root():
    return {
            "mensaje": "API de Riesgo Crediticio activa",
            "docs": "/docs",
            "predict": "/predict"
        }

