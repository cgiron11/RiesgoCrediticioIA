"""Capa HTTP del agente.

Expone `procesar_solicitud()` como endpoint REST para poder correr el agente
como contenedor independiente de la API de prediccion. Es deliberadamente
delgada: toda la logica vive en agent/agent_logic.py.

Levantar:  uvicorn agent.server:app --host 0.0.0.0 --port 8010
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.agent_logic import procesar_solicitud

app = FastAPI(
    title="Agente de Riesgo Crediticio",
    version="1.0.0",
    description="Extrae datos de texto libre, predice el riesgo y justifica la decision.",
)


class SolicitudIn(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        description="Texto libre del solicitante con sus datos financieros",
        examples=[
            "Uso el 45% de mi cupo, debo 2500 dolares, la tasa es 28%, "
            "me consultaron 9 veces, pago 40 dias tarde, mezcla de credito mala, "
            "solo pago el minimo, 30 meses de historial, 18 pagos atrasados "
            "y gano 1500 dolares al mes."
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok", "servicio": "agente"}


@app.post("/solicitud")
def solicitud(entrada: SolicitudIn):
    """Ejecuta el flujo completo y devuelve la evaluacion estructurada."""
    return procesar_solicitud(entrada.texto)
