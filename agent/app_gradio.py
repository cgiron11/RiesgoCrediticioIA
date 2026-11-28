"""Interfaz Gradio del agente de evaluacion de riesgo crediticio.

Uso local:   python agent/app_gradio.py
Docker:      el servicio `agent` de docker-compose ejecuta este mismo archivo.

La interfaz es una capa delgada: solo traduce la llamada a
`procesar_solicitud()` (agent/agent_logic.py) en componentes visuales.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

# Permite ejecutar `python agent/app_gradio.py` desde cualquier carpeta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from agent.agent_logic import procesar_solicitud

TITULO = "Agente de Evaluación de Riesgo Crediticio"
DESCRIPCION = (
    "Escribe en lenguaje natural los datos del cliente (las 10 variables del modelo). "
    "El agente extraerá los datos con un LLM local, los enviará a la API FastAPI, "
    "aplicará la regla de negocio y devolverá la decisión junto con su justificación."
)
PLACEHOLDER = (
    "Ejemplo: Solicito un crédito personal. Uso el 35% de mi cupo, debo 1.400 dólares, "
    "me cobran una tasa de 15%, me consultaron el crédito 5 veces, pago 18 días tarde, "
    "mi mezcla de crédito es estándar, pago más del mínimo, llevo 150 meses de historial, "
    "he tenido 6 pagos atrasados y gano 3.800 dólares al mes netos."
)

# Perfil de riesgo bajo: buenos ingresos, poca deuda y historial limpio.
EJEMPLO_BAJO = (
    "Solicito un crédito con buen perfil. Uso solo el 22% de mi cupo, debo 450 dólares, "
    "me cobran una tasa de 6%, me consultaron el crédito 1 vez, pago siempre a tiempo "
    "(2 días de retraso), mi mezcla de crédito es buena, pago más del mínimo de mis "
    "cuentas, llevo 360 meses de historial, solo he tenido 1 pago atrasado y gano "
    "7.500 dólares al mes netos."
)

# Perfil de riesgo medio: numeros intermedios en casi todas las variables.
EJEMPLO_MEDIO = (
    "Quiero evaluar mi solicitud. Uso el 34% de mi cupo, debo 1.600 dólares, me cobran "
    "una tasa de 16%, me consultaron el crédito 6 veces, pago en promedio 20 días tarde, "
    "mi mezcla de crédito es estándar, pago más del mínimo de mis cuentas, llevo 150 meses "
    "de historial, he tenido 6 pagos atrasados y gano 3.800 dólares al mes netos."
)

# Perfil de riesgo alto: retrasos de pago y alta deuda.
EJEMPLO_ALTO = (
    "Necesito financiamiento urgente. Uso el 47% de mi cupo, debo 3.600 dólares, me "
    "cobran una tasa de 30%, me consultaron el crédito 12 veces, pago 45 días tarde, "
    "mi mezcla de crédito es mala, siempre solo pago el mínimo, llevo 24 meses de "
    "historial, he tenido 22 pagos atrasados y gano 1.200 dólares al mes netos."
)

ESTILOS = {
    "low": {"color": "#1b5e20", "fondo": "#e8f5e9", "titulo": "Riesgo bajo"},
    "medium": {"color": "#e65100", "fondo": "#fff3e0", "titulo": "Riesgo medio"},
    "high": {"color": "#b71c1c", "fondo": "#ffebee", "titulo": "Riesgo alto"},
    "missing": {"color": "#f57f17", "fondo": "#fffde7", "titulo": "Faltan datos"},
    "error": {"color": "#b71c1c", "fondo": "#ffebee", "titulo": "Error"},
}


def _caja(tipo: str, titulo: str, cuerpo: str, detalle: str | None = None) -> str:
    """Construye el cuadro destacado con el color del resultado.

    Los parrafos y spans internos (regla de negocio, score) fuerzan
    color #111827 !important para que sigan siendo legibles en modo oscuro.
    """
    estilo = ESTILOS.get(tipo, ESTILOS["error"])
    detalle_html = (
        f'<p style="margin:6px 0 0; font-size:14px; color:#111827 !important;">'
        f"{detalle}</p>"
        if detalle
        else ""
    )
    return (
        "<style>"
        ".decision-card p, .decision-card span, .decision-card b, "
        ".decision-card strong { color:#111827 !important; }"
        "</style>"
        f'<div class="decision-card" style="border-left:6px solid {estilo["color"]}; '
        f'background:{estilo["fondo"]}; padding:14px 18px; border-radius:8px;">'
        f'<div style="font-size:12px; letter-spacing:.5px; text-transform:uppercase; '
        f'color:#666;">Decisión del Agente</div>'
        f'<div style="font-size:20px; font-weight:700; color:{estilo["color"]}; margin:4px 0;">'
        f"{html.escape(titulo)}</div>"
        f'<p style="margin:0; font-size:15px; color:#111827 !important;">{cuerpo}</p>'
        f"{detalle_html}</div>"
    )


def evaluar(texto: str):
    """Ejecuta el flujo completo del agente y devuelve los 4 bloques de la UI."""
    resultado = procesar_solicitud(texto or "")
    estado = resultado.get("status")
    datos_json = {
        "status": estado,
        "mensaje": resultado.get("mensaje"),
        "datos_extraidos": resultado.get("datos_extraidos"),
        "missing_fields": resultado.get("missing_fields"),
        "advertencias": resultado.get("advertencias"),
    }
    api_json = resultado.get("respuesta_api") or {
        "error": resultado.get("mensaje", "Sin respuesta de la API")
    }

    if estado == "ok":
        prediccion = resultado.get("respuesta_api") or {}
        nivel = prediccion.get("risk_level", "medium")
        cuerpo = (
            f"<b>{html.escape(str(resultado.get('decision', '')))}</b>"
        )
        detalle = (
            f"Nivel de riesgo: <b>{html.escape(str(nivel))}</b> "
            f"(código {prediccion.get('risk_code')}) &nbsp;·&nbsp; "
            f"Score: <b>{prediccion.get('score')}</b>"
        )
        decision = _caja(nivel, ESTILOS[nivel]["titulo"], cuerpo, detalle)
        justificacion = resultado.get("justificacion") or "_Sin justificación disponible._"
    elif estado == "missing_fields":
        faltan = ", ".join(resultado.get("missing_fields", []))
        decision = _caja(
            "missing",
            "Faltan datos obligatorios",
            html.escape(resultado.get("mensaje", "")),
            f"Campos pendientes: <b>{html.escape(faltan)}</b>",
        )
        justificacion = (
            "_Completa los campos indicados y vuelve a pulsar "
            "**Evaluar solicitud** para obtener la decisión._"
        )
    else:
        decision = _caja(
            "error",
            "No se pudo evaluar la solicitud",
            html.escape(resultado.get("mensaje", "Error desconocido")),
            "Revisa que la API y Ollama estén en marcha.",
        )
        justificacion = "_Sin justificación: corrige el error y vuelve a intentarlo._"

    return decision, justificacion, datos_json, api_json


with gr.Blocks(title=TITULO) as demo:
    gr.Markdown(f"# {TITULO}")
    gr.Markdown(DESCRIPCION)

    with gr.Row():
        entrada = gr.Textbox(
            label="Solicitud del cliente (lenguaje natural)",
            placeholder=PLACEHOLDER,
            lines=7,
            max_lines=14,
        )

    boton = gr.Button("Evaluar solicitud", variant="primary")

    gr.Examples(
        examples=[[EJEMPLO_BAJO], [EJEMPLO_MEDIO], [EJEMPLO_ALTO]],
        inputs=[entrada],
        label="Ejemplos precargados (haz clic para probar)",
        examples_per_page=3,
    )

    with gr.Row():
        with gr.Column(scale=1):
            salida_decision = gr.HTML(label="Decisión del Agente")
        with gr.Column(scale=1):
            gr.Markdown("### Justificación Detallada")
            salida_justificacion = gr.Markdown()

    with gr.Accordion(
        "Detalle técnico: JSON extraído y respuesta de la API", open=False
    ):
        salida_datos = gr.JSON(label="JSON extraído por el LLM (10 características)")
        salida_api = gr.JSON(label="Respuesta cruda de la API FastAPI (/predict)")

    gr.Markdown(
        "_Pipeline: Ollama (extracción) → FastAPI `/predict` → regla de negocio "
        "→ Ollama (justificación)._"
    )

    boton.click(
        fn=evaluar,
        inputs=[entrada],
        outputs=[salida_decision, salida_justificacion, salida_datos, salida_api],
    )
    entrada.submit(
        fn=evaluar,
        inputs=[entrada],
        outputs=[salida_decision, salida_justificacion, salida_datos, salida_api],
    )


if __name__ == "__main__":
    demo.queue(max_size=8)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1100px; }",
    )
