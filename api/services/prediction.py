"""Servicio de prediccion de riesgo crediticio.

Orquesta todo el camino de inferencia:
  peticion -> vector en el orden de columnas.joblib (categorias ya mapeadas)
  -> DecisionTreeClassifier -> clase + probabilidades -> top 3 factores.

El arbol no requiere escalado ni codificador: las categorias llegan mapeadas
igual que en el notebook (Credit_Mix 0=Bad/1=Standard/2=Good y
Payment_of_Min_Amount 0=No/1=NM/2=Yes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from api.models.artifacts import ARTIFACTS, ArtifactStore
from api.schemas import CreditRequest

# Mapa interno de la salida del modelo a los niveles de riesgo de la API.
NIVELES_RIESGO = {0: "low", 1: "medium", 2: "high"}

# Mapeo identico al del notebook para las dos caracteristicas categoricas.
MAPEO_CATEGORIAS: dict[str, dict[str, int]] = {
    "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2},
    "Payment_of_Min_Amount": {"No": 0, "NM": 1, "Yes": 2},
}

# Numero de factores devueltos: las 3 variables de mayor feature_importances_.
TOP_N = 3


class PredictionError(Exception):
    """Fallo durante la inferencia (artefactos, vector o modelo)."""


class PredictionService:
    """Transforma una CreditRequest en la prediccion estructurada."""

    def __init__(self, store: ArtifactStore = ARTIFACTS) -> None:
        self.store = store

    def predict(self, peticion: CreditRequest) -> dict[str, Any]:
        """Devuelve {risk_code, risk_level, score, top_factors}."""
        try:
            vector = self._vector(peticion)
            probas = self._probabilidades(vector)
            clase = int(np.argmax(probas))
            if clase not in NIVELES_RIESGO:
                raise PredictionError(f"Clase fuera de rango: {clase}")
            return {
                "risk_code": clase,
                "risk_level": NIVELES_RIESGO[clase],
                "score": round(float(np.max(probas)), 4),
                "top_factors": self._top_factors(peticion),
            }
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Error en la inferencia: {exc}") from exc

    def _vector(self, peticion: CreditRequest) -> np.ndarray:
        """Fila en el orden exacto de columnas.joblib; categorias ya mapeadas."""
        valores: list[float] = []
        for nombre in self.store.feature_order:
            if nombre in MAPEO_CATEGORIAS:
                bruto = str(getattr(peticion, nombre))
                if bruto not in MAPEO_CATEGORIAS[nombre]:
                    raise PredictionError(
                        f"Valor categorico no mapeado para '{nombre}': {bruto}"
                    )
                valores.append(float(MAPEO_CATEGORIAS[nombre][bruto]))
            else:
                valores.append(float(getattr(peticion, nombre)))
        return np.array([valores], dtype=float)

    def _probabilidades(self, vector: np.ndarray) -> np.ndarray:
        """Probabilidades por clase; fallback a softmax si el modelo no expone predict_proba."""
        modelo = self.store.model
        if hasattr(modelo, "predict_proba"):
            return np.asarray(modelo.predict_proba(vector)[0], dtype=float)
        if hasattr(modelo, "decision_function"):
            logits = np.asarray(modelo.decision_function(vector), dtype=float)[0]
            exponencial = np.exp(logits - logits.max())
            return exponencial / exponencial.sum()
        raise PredictionError("El modelo no expone probabilidades")

    def _top_factors(self, peticion: CreditRequest) -> list[dict[str, Any]]:
        """Las 3 variables de mayor feature_importances_ con el valor del cliente."""
        importancias = self.store.importancias
        ranking = sorted(importancias.items(), key=lambda par: par[1], reverse=True)
        return [
            {
                "feature": nombre,
                "value": getattr(peticion, nombre),
                "weight_pct": round(peso * 100.0, 2),
            }
            for nombre, peso in ranking[:TOP_N]
        ]
