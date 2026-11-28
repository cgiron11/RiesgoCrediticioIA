"""Modelos de datos de la API: peticion, respuesta y health."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreditRequest(BaseModel):
    """Las 10 caracteristicas exactas usadas en el entrenamiento."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Credit_Utilization_Ratio": 32.5,
                "Outstanding_Debt": 1200.75,
                "Interest_Rate": 14,
                "Num_Credit_Inquiries": 4,
                "Delay_from_due_date": 12,
                "Credit_Mix": "Standard",
                "Payment_of_Min_Amount": "No",
                "Credit_History_Age": 240,
                "Num_of_Delayed_Payment": 8,
                "Monthly_Inhand_Salary": 4200.5,
            }
        }
    )

    Credit_Utilization_Ratio: float = Field(
        ..., ge=0, le=100, description="Porcentaje de uso del credito (0-100)"
    )
    Outstanding_Debt: float = Field(..., ge=0, description="Deuda pendiente en USD")
    Interest_Rate: float = Field(..., ge=0, le=100, description="Tasa de interes (%)")
    Num_Credit_Inquiries: int = Field(..., ge=0, le=50, description="Consultas de credito")
    Delay_from_due_date: int = Field(
        ..., ge=-30, le=365, description="Dias de retraso respecto al vencimiento"
    )
    Credit_Mix: Literal["Bad", "Standard", "Good"] = Field(
        ..., description="Mezcla de tipos de credito"
    )
    Payment_of_Min_Amount: Literal["No", "NM", "Yes"] = Field(
        ..., description="Si solo pago el minimo: No / NM (no mencionado) / Yes"
    )
    Credit_History_Age: float = Field(
        ..., ge=0, le=1200, description="Antiguedad del historial de credito en meses"
    )
    Num_of_Delayed_Payment: int = Field(..., ge=0, le=100, description="Pagos atrasados")
    Monthly_Inhand_Salary: float = Field(..., ge=0, description="Salario mensual neto USD")


class TopFactor(BaseModel):
    """Variable con mayor peso en el arbol de decision, con el valor del cliente."""

    feature: str = Field(..., description="Nombre de la caracteristica")
    value: str | float = Field(..., description="Valor real ingresado por el cliente")
    weight_pct: float = Field(
        ...,
        description="Peso porcentual de la variable en feature_importances_ (0-100)",
    )


class CreditResponse(BaseModel):
    """Salida del endpoint /predict."""

    risk_code: int = Field(..., ge=0, le=2, description="0=bajo, 1=medio, 2=alto")
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Nivel de riesgo en texto"
    )
    score: float = Field(
        ..., ge=0, le=1, description="Probabilidad maxima redondeada a 4 decimales"
    )
    top_factors: list[TopFactor] = Field(
        ...,
        description="3 variables con mayor feature_importances_ del arbol, con el valor del cliente y su peso %",
        max_length=3,
    )


class HealthResponse(BaseModel):
    """Salida del endpoint /health."""

    status: Literal["ok", "error"]
    model_loaded: bool
    model_name: str | None = None
    n_features: int = 0
    detail: str | None = None
