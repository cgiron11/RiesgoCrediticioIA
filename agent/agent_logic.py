"""Logica del agente de riesgo crediticio.

Flujo de procesar_solicitud():
  1. Extraer las 10 caracteristicas del texto libre con el LLM local (Ollama).
  2. Enviar las caracteristicas a la API REST de prediccion (FastAPI).
  3. Aplicar la regla de negocio obligatoria segun el nivel de riesgo.
  4. Generar la justificacion en espanol con el LLM.
  5. Devolver un diccionario estructurado con todos los resultados.

Variables de entorno:
  OLLAMA_URL     -> http://localhost:11434   (servidor Ollama)
  OLLAMA_MODEL   -> phi3                     (modelo a usar)
  API_URL        -> http://localhost:8000    (API de prediccion)
  OLLAMA_TIMEOUT -> 120                      (segundos)
  API_TIMEOUT    -> 30                       (segundos)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from agent.prompts import (
    ETIQUETAS_RIESGO,
    cargar_metadata,
    construir_extraction_prompt,
    construir_justificacion_prompt,
    tipo_dato,
    valores_permitidos,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
API_URL = os.getenv("API_URL", "http://localhost:8000")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))

# Limite de la justificacion final: 3 oraciones (num_predict=140 en Ollama).
MAX_ORACIONES_JUSTIFICACION = 3
NUM_PREDICT_JUSTIFICACION = 140

# Reglas de negocio obligatorias: son deterministas, el LLM no interviene.
REGLAS_NEGOCIO: dict[str, str] = {
    "high": "Rechazar solicitud y recomendar educacion financiera.",
    "medium": "Solicitar documentacion adicional y evaluar nuevamente.",
    "low": "Aprobar solicitud con condiciones estandar.",
}


class ErrorAgente(Exception):
    """Error base del agente."""


class ErrorOllama(ErrorAgente):
    """Ollama no responde o devuelve algo inesperado."""


class ErrorAPI(ErrorAgente):
    """La API de prediccion no esta disponible o devolvio un error."""


def llamar_ollama(
    prompt: str,
    modelo: str | None = None,
    url: str | None = None,
    timeout: float | None = None,
    num_predict: int | None = None,
) -> str:
    """Invoca /api/generate de Ollama y devuelve solo el texto generado."""
    base = (url or OLLAMA_URL).rstrip("/")
    opciones: dict[str, Any] = {"temperature": 0.0}
    if num_predict is not None:
        opciones["num_predict"] = int(num_predict)
    payload = {
        "model": modelo or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": opciones,
    }
    try:
        respuesta = requests.post(
            f"{base}/api/generate",
            json=payload,
            timeout=timeout or OLLAMA_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ErrorOllama(
            f"No se pudo conectar a Ollama en {base}. "
            f"Verifica que este corriendo ('ollama serve') y que el modelo "
            f"'{modelo or OLLAMA_MODEL}' este descargado."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ErrorOllama(
            f"Ollama no respondio en {timeout or OLLAMA_TIMEOUT}s con el modelo "
            f"'{modelo or OLLAMA_MODEL}'."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ErrorOllama(f"Error inesperado llamando a Ollama: {exc}") from exc

    if respuesta.status_code == 404:
        raise ErrorOllama(
            f"El modelo '{modelo or OLLAMA_MODEL}' no existe en {base}. "
            "Descargalo con: ollama pull qwen2.5-coder:3b"
        )
    if not respuesta.ok:
        raise ErrorOllama(
            f"Ollama respondio HTTP {respuesta.status_code}: {respuesta.text[:300]}"
        )

    try:
        datos = respuesta.json()
    except ValueError as exc:
        raise ErrorOllama("Ollama devolvio una respuesta que no es JSON.") from exc

    texto = str(datos.get("response", "")).strip()
    if not texto:
        raise ErrorOllama("Ollama devolvio una respuesta vacia.")
    return texto


def extraer_json(texto_llm: str) -> dict[str, Any]:
    """Acepta el texto del LLM y devuelve el primer objeto JSON valido."""
    limpio = texto_llm.strip()
    limpio = re.sub(r"^```[a-zA-Z]*\s*", "", limpio)
    limpio = re.sub(r"\s*```$", "", limpio)
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio == -1 or fin <= inicio:
        raise ValueError("el LLM no devolvio ningun objeto JSON")
    return json.loads(limpio[inicio : fin + 1])


def _coercion(valor: Any, nombre: str, metadata: dict[str, Any]) -> Any:
    """Convierte un valor al tipo que exige la API; devuelve None si no se puede."""
    if valor is None:
        return None
    tipo = tipo_dato(nombre)

    if tipo == "texto":
        texto = str(valor).strip()
        permitidos = valores_permitidos(nombre, metadata)
        if permitidos:
            equivalencias = {p.lower(): p for p in permitidos}
            if texto.lower() in equivalencias:
                return equivalencias[texto.lower()]
            return None
        return texto or None

    if isinstance(valor, str):
        valor = valor.strip().replace(" ", "").replace(",", ".")
        valor = re.sub(r"[^0-9.\-]", "", valor)
        if valor in {"", "-", "."}:
            return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    if tipo == "entero":
        return int(round(numero))
    return round(numero, 6)


def extraer_caracteristicas(
    texto_usuario: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Pide al LLM las 10 caracteristicas y normaliza sus valores.

    Devuelve: {"datos": dict, "missing_fields": list, "crudo": str}.
    """
    metadata = metadata or cargar_metadata()
    nombres = metadata.get("caracteristicas_ordenadas", [])
    prompt = construir_extraction_prompt(texto_usuario, metadata)

    ultimo_error: Exception | None = None
    datos_crudos: dict[str, Any] | None = None
    for intento in range(2):
        texto_llm = llamar_ollama(prompt)
        try:
            datos_crudos = extraer_json(texto_llm)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            ultimo_error = exc
            prompt = (
                construir_extraction_prompt(texto_usuario, metadata)
                + "\n\nATENCION: tu respuesta anterior no era un JSON valido. "
                "Responde SOLO con el objeto JSON, sin ningun texto adicional."
            )
    if datos_crudos is None:
        raise ErrorOllama(f"El LLM no devolvio un JSON valido ({ultimo_error}).")

    declarados = {
        str(f).strip()
        for f in datos_crudos.get("missing_fields", [])
        if isinstance(f, str) and str(f).strip()
    }

    datos: dict[str, Any] = {}
    faltantes: list[str] = []
    for nombre in nombres:
        convertido = _coercion(datos_crudos.get(nombre), nombre, metadata)
        if convertido is None:
            faltantes.append(nombre)
        else:
            datos[nombre] = convertido

    # Si el LLM entrego un valor valido, manda sobre su propia lista de faltantes.
    for nombre in declarados:
        if nombre in nombres and nombre not in datos and nombre not in faltantes:
            faltantes.append(nombre)

    desconocidas = [
        clave
        for clave in datos_crudos.keys()
        if clave not in nombres and clave != "missing_fields"
    ]
    return {
        "datos": datos,
        "missing_fields": [n for n in nombres if n in faltantes],
        "crudo": json.dumps(datos_crudos, ensure_ascii=False),
        "claves_desconocidas": desconocidas,
    }


def consultar_api(datos: dict[str, Any], url: str | None = None) -> dict[str, Any]:
    """POST /predict con las 10 caracteristicas; maneja ConnectionError/Timeout/422/500."""
    base = (url or API_URL).rstrip("/")
    try:
        respuesta = requests.post(
            f"{base}/predict",
            json=datos,
            timeout=API_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ErrorAPI(
            f"No se pudo conectar a la API de prediccion en {base}. "
            "Levantala con: uvicorn api.main:app --port 8000"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ErrorAPI(
            f"La API de prediccion no respondio en {API_TIMEOUT}s ({base})."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ErrorAPI(f"Error inesperado llamando a la API: {exc}") from exc

    if respuesta.status_code == 422:
        raise ErrorAPI(_mensaje_422(respuesta))
    if respuesta.status_code >= 500:
        detalle = _detalle(respuesta)
        raise ErrorAPI(f"La API fallo con HTTP {respuesta.status_code}: {detalle}")
    if not respuesta.ok:
        raise ErrorAPI(
            f"La API respondio HTTP {respuesta.status_code}: {_detalle(respuesta)}"
        )

    try:
        return respuesta.json()
    except ValueError as exc:
        raise ErrorAPI("La API devolvio una respuesta que no es JSON.") from exc


def _detalle(respuesta: requests.Response) -> str:
    try:
        datos = respuesta.json()
    except ValueError:
        return respuesta.text[:300]
    if isinstance(datos, dict):
        return str(datos.get("detail", datos))[:300]
    return str(datos)[:300]


def _mensaje_422(respuesta: requests.Response) -> str:
    """Traduce la lista de errores de validacion de Pydantic a un mensaje legible."""
    try:
        detalle = respuesta.json().get("detail", [])
    except ValueError:
        return "La API rechazo la peticion (HTTP 422)."
    errores = []
    for error in detalle if isinstance(detalle, list) else [detalle]:
        if isinstance(error, dict):
            campo = ".".join(str(p) for p in error.get("loc", []))
            errores.append(f"{campo}: {error.get('msg', 'valor invalido')}")
        else:
            errores.append(str(error))
    return "Datos invalidos para la API (HTTP 422): " + "; ".join(errores)


def aplicar_regla(risk_level: str) -> str:
    """Devuelve la decision oficial obligatoria para un nivel de riesgo."""
    if risk_level not in REGLAS_NEGOCIO:
        raise ErrorAPI(f"Nivel de riesgo desconocido: '{risk_level}'.")
    return REGLAS_NEGOCIO[risk_level]


def _limpiar_justificacion(texto: str) -> str:
    """Quita markdown, numeraciones y saltos de linea que el LLM pueda colar.

    Conserva los guiones bajos dentro de los nombres de variable
    (p. ej. Outstanding_Debt) y solo elimina los marcadores de cursiva.
    """
    texto = re.sub(r"(__|\*\*|`+)", "", texto)
    texto = re.sub(r"(?<![\w])_(?=\w)|(?<=\w)_(?![\w])", "", texto)
    texto = re.sub(r"(?m)^\s*#{1,6}\s*", "", texto)
    texto = re.sub(r"(?m)^\s*\d+[.)]\s*", "", texto)
    texto = re.sub(r"(?m)^\s*[-•]\s*", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.replace('"', "").strip()


def _asegurar_etiqueta(texto: str, etiqueta: str) -> str:
    """Fuerza que el texto empiece exactamente con la etiqueta oficial."""
    limpio = re.sub(
        r"^(?:riesgo\s+(?:bajo|medio|alto))\s*[:\-–]?\s*",
        "",
        texto.strip(),
        flags=re.IGNORECASE,
    )
    return f"{etiqueta}: {limpio}" if limpio else etiqueta


def _limitar_oraciones(texto: str, max_oraciones: int = MAX_ORACIONES_JUSTIFICACION) -> str:
    """Recorta el texto a como maximo N oraciones."""
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+", texto.strip()) if o]
    return " ".join(oraciones[:max_oraciones]).strip()


def _faltan_pesos(texto: str, prediccion: dict[str, Any]) -> list[str]:
    """Devuelve los pesos % (en texto) que la justificacion no cita."""
    faltan = []
    for factor in prediccion.get("top_factors") or []:
        peso = factor.get("weight_pct")
        if peso is None:
            continue
        opciones = {str(peso), str(peso).replace(".", ",")}
        if not any(opcion in texto for opcion in opciones):
            faltan.append(str(peso))
    return faltan


def generar_justificacion(
    prediccion: dict[str, Any],
    regla: str,
) -> str:
    """Pide al LLM la justificacion en espanol y la normaliza.

    Garantiza tres invariantes exigidos por el prompt:
      1. empieza exactamente con la etiqueta oficial (Riesgo bajo/medio/alto);
      2. menciona solo top_factors (con sus pesos %) y la regla de negocio;
      3. no supera las 3 oraciones (num_predict=140 en Ollama).

    Si el LLM omite algun porcentaje, se repite una sola vez con un recordatorio.
    """
    etiqueta = ETIQUETAS_RIESGO.get(
        str(prediccion.get("risk_level", "")), "Riesgo medio"
    )
    prompt = construir_justificacion_prompt(prediccion, regla)
    justificacion = _limpiar_justificacion(
        llamar_ollama(prompt, num_predict=NUM_PREDICT_JUSTIFICACION)
    )

    if _faltan_pesos(justificacion, prediccion):
        justificacion = _limpiar_justificacion(
            llamar_ollama(
                prompt
                + "\n\nATENCION: tu respuesta anterior no citaba los porcentajes de "
                "los 3 factores. Redactala otra vez citando obligatoriamente los 3 "
                "porcentajes y la regla textual, con el mismo formato.",
                num_predict=NUM_PREDICT_JUSTIFICACION,
            )
        )

    justificacion = _asegurar_etiqueta(justificacion, etiqueta)
    return _limitar_oraciones(justificacion, MAX_ORACIONES_JUSTIFICACION)


def procesar_solicitud(texto_usuario: str) -> dict[str, Any]:
    """Ejecuta el flujo completo: extraccion -> prediccion -> regla -> justificacion."""
    resultado: dict[str, Any] = {
        "status": "ok",
        "mensaje": "",
        "datos_extraidos": None,
        "missing_fields": [],
        "respuesta_api": None,
        "decision": None,
        "regla_aplicada": None,
        "justificacion": None,
        "advertencias": [],
    }

    if not texto_usuario or not texto_usuario.strip():
        resultado.update(
            status="error",
            mensaje="El texto de la solicitud esta vacio.",
        )
        return resultado

    try:
        metadata = cargar_metadata()
    except (OSError, ValueError) as exc:
        resultado.update(status="error", mensaje=f"No se pudo leer metadata.json: {exc}")
        return resultado

    # 1) Extraccion de las 10 caracteristicas con el LLM.
    try:
        extraccion = extraer_caracteristicas(texto_usuario, metadata)
    except ErrorAgente as exc:
        resultado.update(status="error", mensaje=str(exc))
        return resultado

    faltantes = extraccion["missing_fields"]
    if faltantes:
        resultado.update(
            status="missing_fields",
            datos_extraidos=extraccion["datos"],
            missing_fields=faltantes,
            mensaje=(
                "Faltan datos obligatorios para evaluar la solicitud: "
                + ", ".join(faltantes)
                + ". Por favor, agrega esa informacion y vuelve a intentar."
            ),
        )
        return resultado

    if extraccion["claves_desconocidas"]:
        resultado["advertencias"].append(
            "Se ignoraron claves no reconocidas: "
            + ", ".join(extraccion["claves_desconocidas"])
        )

    datos = extraccion["datos"]
    resultado["datos_extraidos"] = datos

    # 2) Prediccion con la API REST.
    try:
        prediccion = consultar_api(datos)
    except ErrorAPI as exc:
        resultado.update(status="error", mensaje=str(exc))
        return resultado

    resultado["respuesta_api"] = prediccion
    risk_level = prediccion.get("risk_level")

    # 3) Regla de negocio deterministica.
    try:
        regla = aplicar_regla(risk_level)
    except ErrorAPI as exc:
        resultado.update(status="error", mensaje=str(exc))
        return resultado
    resultado["decision"] = regla
    resultado["regla_aplicada"] = regla

    # 4) Justificacion tecnica con el LLM (no bloquea la decision si falla).
    try:
        resultado["justificacion"] = generar_justificacion(prediccion, regla)
    except ErrorAgente as exc:
        resultado["justificacion"] = (
            f"Decision tomada por regla de negocio sin justificacion del LLM: {regla}"
        )
        resultado["advertencias"].append(f"Justificacion no disponible: {exc}")

    resultado["mensaje"] = "Solicitud evaluada correctamente."
    return resultado
