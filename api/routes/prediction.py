"""Ruta POST /predict."""

from fastapi import APIRouter

from api.controllers import PredictionController
from api.schemas import CreditRequest, CreditResponse

router = APIRouter(tags=["prediccion"])
controller = PredictionController()


@router.post("/predict", response_model=CreditResponse)
def predict(peticion: CreditRequest) -> CreditResponse:
    """Recibe las 10 caracteristicas y devuelve el nivel de riesgo."""
    return controller.predict(peticion)
