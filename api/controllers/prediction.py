"""Controladores de los endpoints de riesgo crediticio.

No contienen logica de ML: delegan en PredictionService y se encargan unicamente
del mapeo a codigos HTTP (200 / 500) y a los esquemas de respuesta.
"""

from __future__ import annotations

from fastapi import HTTPException

from api.models.artifacts import ARTIFACTS, ArtifactError
from api.schemas import CreditRequest, CreditResponse, HealthResponse
from api.services.prediction import PredictionError, PredictionService


class PredictionController:
    """Expone las operaciones de prediccion y salud para las rutas."""

    def __init__(self, service: PredictionService | None = None) -> None:
        self.service = service or PredictionService()

    def health(self) -> HealthResponse:
        """Estado del modelo cargado en memoria."""
        if not ARTIFACTS.loaded:
            return HealthResponse(
                status="error",
                model_loaded=False,
                detail=ARTIFACTS.error or "Artefactos no encontrados",
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=ARTIFACTS.model_name,
            n_features=len(ARTIFACTS.feature_order),
        )

    def predict(self, peticion: CreditRequest) -> CreditResponse:
        """Ejecuta la inferencia o lanza un HTTP 500 descriptivo."""
        try:
            resultado = self.service.predict(peticion)
            return CreditResponse(**resultado)
        except (PredictionError, ArtifactError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Error en la inferencia: {exc}"
            ) from exc
