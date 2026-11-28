"""Carga y cachea los artefactos exportados por el notebook.

El notebook `data/entrenamiento_riesgo.ipynb` exporta tres archivos en models/:

  modelo_riesgo.joblib   -> DecisionTreeClassifier (sin escalado; el arbol no lo usa)
  columnas.joblib        -> list(X.columns): orden exacto de las 10 caracteristicas
  importancias.joblib    -> {columna: feature_importances_} del arbol

Las categorias llegan ya mapeadas a numeros, igual que en el notebook:
  Credit_Mix             0=Bad, 1=Standard, 2=Good
  Payment_of_Min_Amount  0=No, 1=NM, 2=Yes

El almacen es un singleton (ARTIFACTS) que se carga una sola vez al arrancar la
aplicacion; el resto de capas solo lo consultan, nunca leen el disco directamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = BASE_DIR / "models"


class ArtifactError(Exception):
    """Los artefactos no estan disponibles o estan corruptos."""


class ArtifactStore:
    """Repositorio de artefactos de ML con carga perezosa y cache en memoria."""

    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.error: str | None = None
        self._model: Any = None
        self._columnas: list[str] | None = None
        self._importancias: dict[str, float] | None = None

    def load(self) -> bool:
        """Lee los 3 artefactos del notebook en models/. Devuelve False si algo falla."""
        try:
            self._model = joblib.load(self.models_dir / "modelo_riesgo.joblib")
            columnas = joblib.load(self.models_dir / "columnas.joblib")
            importancias = joblib.load(self.models_dir / "importancias.joblib")
            self._columnas = [str(c) for c in columnas]
            self._importancias = {str(k): float(v) for k, v in importancias.items()}
            self.error = None
            return True
        except Exception as exc:  # archivo ausente, joblib corrupto
            self.error = str(exc)
            return False

    @property
    def loaded(self) -> bool:
        return all(
            pieza is not None
            for pieza in (self._model, self._columnas, self._importancias)
        )

    @property
    def model(self) -> Any:
        return self._require(self._model, "modelo")

    @property
    def columnas(self) -> list[str]:
        """Orden exacto de las 10 caracteristicas usado en el entrenamiento."""
        return list(self._require(self._columnas, "columnas"))

    @property
    def importancias(self) -> dict[str, float]:
        """feature_importances_ del arbol por nombre de columna."""
        return dict(self._require(self._importancias, "importancias"))

    def _require(self, pieza: Any, nombre: str) -> Any:
        if pieza is None:
            raise ArtifactError(self.error or f"Artefacto '{nombre}' no cargado")
        return pieza

    @property
    def model_name(self) -> str | None:
        if self._model is None:
            return None
        return type(self._model).__name__

    @property
    def feature_order(self) -> list[str]:
        """Alias usado por servicios y controladores."""
        return self.columnas


# Singleton usado por servicios y controladores.
ARTIFACTS = ArtifactStore()
