"""Prompts del agente de riesgo crediticio.

Este módulo define las dos plantillas de prompt que se envían al LLM local
(Ollama) y las funciones que las rellenan con informacion "viva" leida desde
models/metadata.json: orden, tipo, descripcion, rango tipico y categorias de
las 10 caracteristicas usadas en el entrenamiento.

Asi, si se reentrena el modelo y cambian los rangos o las categorias, el agente
se actualiza solo sin tocar este archivo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, get_origin

BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_METADATA = BASE_DIR / "models" / "metadata.json"

# Descripciones en espanol de cada caracteristica. Si metadata.json trae la
# clave "descripcion", se usa esa; este diccionario es el respaldo.
DESCRIPCIONES: dict[str, str] = {
    "Credit_Utilization_Ratio": "Porcentaje de uso del credito respecto al limite disponible (0-100)",
    "Outstanding_Debt": "Deuda pendiente total del solicitante en USD",
    "Interest_Rate": "Tasa de interes promedio (%) que paga el solicitante",
    "Num_Credit_Inquiries": "Numero de consultas de credito en su historial reciente",
    "Delay_from_due_date": "Dias promedio de retraso respecto a la fecha de vencimiento",
    "Credit_Mix": "Diversidad de tipos de credito que maneja (Bad / Standard / Good)",
    "Payment_of_Min_Amount": "Si paga solo el minimo de sus cuentas: Yes = solo paga el minimo, No = paga mas del minimo, NM = no mencionado",
    "Credit_History_Age": "Antiguedad del historial de credito en meses",
    "Num_of_Delayed_Payment": "Numero de pagos atrasados registrados",
    "Monthly_Inhand_Salary": "Salario mensual neto percibido en USD",
}

# Tipos esperados por la API. Resaldo por si api.schemas no se puede importar.
TIPOS: dict[str, str] = {
    "Credit_Utilization_Ratio": "decimal",
    "Outstanding_Debt": "decimal",
    "Interest_Rate": "decimal",
    "Num_Credit_Inquiries": "entero",
    "Delay_from_due_date": "entero",
    "Credit_Mix": "texto",
    "Payment_of_Min_Amount": "texto",
    "Credit_History_Age": "decimal",
    "Num_of_Delayed_Payment": "entero",
    "Monthly_Inhand_Salary": "decimal",
}

# Plantilla de extraccion: recibe el texto libre del usuario y el listado de
# variables ya descritas. El LLM debe responder SOLO con el JSON.
EXTRACTION_PROMPT = """\
Eres un motor de extraccion de datos para un sistema de evaluacion de riesgo crediticio.
Tu unica tarea es convertir el texto libre del solicitante en un objeto JSON.

REGLAS OBLIGATORIAS:
1. Responde UNICAMENTE con un objeto JSON valido: sin markdown, sin bloques de codigo, sin comentarios y sin texto antes o despues.
2. El objeto debe tener EXACTAMENTE las 10 claves de VARIABLES REQUERIDAS mas la clave "missing_fields". No agregues claves extra.
3. Respeta los tipos: "entero" sin decimales, "decimal" con punto decimal y sin separador de miles, "texto" entre comillas dobles.
4. Nunca inventes datos. Si el texto no aporta un valor para alguna variable, asigna null a esa clave y anade el nombre exacto de la clave al arreglo "missing_fields".
5. "missing_fields" es siempre un arreglo de cadenas; usa [] cuando el texto contenga las 10 variables.
6. Normaliza unidades y formatos: antiguedad del historial en anos/meses -> meses; importes con coma decimal -> punto decimal; montos con simbolo de moneda -> solo el numero.
7. Para las variables categoricas usa EXACTAMENTE uno de los valores permitidos (traduce sinonimos en espanol al valor permitido).
8. Semantica que suele confundirse:
   - Payment_of_Min_Amount: SI el texto dice que paga solo o unicamente el minimo, en
     cualquier orden de palabras (p. ej. "unicamente pago el minimo", "solo pago el minimo",
     "pago el minimo de mis tarjetas") -> Yes. SI dice que paga mas del minimo o el pago
     completo -> No. Si no lo menciona -> null y anade la clave a missing_fields.
   - Credit_Mix: "mala/deficiente" -> Bad; "regular/estandar" -> Standard; "buena/diversa" -> Good.

EJEMPLO (solo referencia de formato, no lo repitas):
Texto: "Uso el 30% de mi cupo, debo 900 dolares, me cobran una tasa de 12%, me consultaron 2 veces, pago 5 dias tarde, mi mezcla de credito es buena, llevo 120 meses de historial, he tenido 2 pagos atrasados y gano 4000 dolares al mes."
Salida: {{"Credit_Utilization_Ratio": 30.0, "Outstanding_Debt": 900.0, "Interest_Rate": 12.0, "Num_Credit_Inquiries": 2, "Delay_from_due_date": 5, "Credit_Mix": "Good", "Payment_of_Min_Amount": null, "Credit_History_Age": 120.0, "Num_of_Delayed_Payment": 2, "Monthly_Inhand_Salary": 4000.0, "missing_fields": ["Payment_of_Min_Amount"]}}

VARIABLES REQUERIDAS:
{caracteristicas}

TEXTO DEL SOLICITANTE:
<<<
{texto_usuario}
>>>

JSON de salida:"""

# Plantilla de justificacion: recibe la prediccion de la API (con sus
# top_factors y pesos %) y la regla de negocio aplicada. El LLM solo redacta;
# no aporta datos ni comparaciones propias.
JUSTIFICATION_PROMPT = """\
Eres un analista de riesgo crediticio de un banco.
Redacta en espanol la explicacion de la decision de prestamo.

REGLAS ESTRICTAS:
1. La respuesta debe empezar EXACTAMENTE con la etiqueta oficial "{etiqueta}"
   (por ejemplo: "{etiqueta}: el solicitante ..."). No empieces con ninguna otra frase.
2. Maximo 3 oraciones en total, en espanol profesional y claro.
3. PROHIBIDO: mencionar cualquier otro nivel de riesgo distinto de "{etiqueta}",
   promedios o medias del dataset, probabilidades/score, cifras inventadas o datos
   que no figuren en los FACTORES DESTACADOS.
4. Es OBLIGATORIO citar los 3 FACTORES DESTACADOS con su porcentaje tal cual aparece
   (por ejemplo: "Outstanding_Debt 39.17%"), usando unicamente esos datos.
5. Es OBLIGATORIO cerrar citando textualmente la REGLA DE NEGOCIO APLICADA.
6. Sin markdown, sin vinetas, sin numeracion y sin saltos de linea.

FACTORES DESTACADOS (debes citar los 3 con su peso %):
{factores}

REGLA DE NEGOCIO APLICADA (citate textualmente al final):
{regla}

ETIQUETA OFICIAL A USAR AL INICIO: {etiqueta}

Explicacion:"""


def cargar_metadata(ruta: Path | str = RUTA_METADATA) -> dict[str, Any]:
    """Lee models/metadata.json (caracteristicas, medias, rangos y clases)."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def _campos_api() -> dict[str, Any]:
    """Obtiene los campos de CreditRequest para conocer los tipos reales."""
    try:
        from api.schemas import CreditRequest

        return dict(CreditRequest.model_fields)
    except Exception:
        return {}


def tipo_dato(nombre: str) -> str:
    """Devuelve 'entero', 'decimal' o 'texto' para una caracteristica."""
    campos = _campos_api()
    anotacion = campos.get(nombre).annotation if nombre in campos else None
    if anotacion is int:
        return "entero"
    if anotacion is float:
        return "decimal"
    if anotacion is str:
        return "texto"
    if anotacion is not None and get_origin(anotacion) is Literal:
        return "texto"
    return TIPOS.get(nombre, "decimal")


def valores_permitidos(nombre: str, metadata: dict[str, Any]) -> list[str]:
    """Lista de categorias permitidas (vacio si la caracteristica es numerica)."""
    for entrada in metadata.get("caracteristicas", []):
        if entrada.get("nombre") == nombre:
            return list(entrada.get("categorias", {}).keys())
    return []


def _descriptor(nombre: str, metadata: dict[str, Any]) -> str:
    """Construye la linea descriptiva de una caracteristica para el prompt."""
    entrada = next(
        (c for c in metadata.get("caracteristicas", []) if c.get("nombre") == nombre),
        {},
    )
    descripcion = entrada.get("descripcion") or DESCRIPCIONES.get(nombre, "")
    tipo = tipo_dato(nombre)
    partes = [f"- {nombre}", f"tipo: {tipo}", f"descripcion: {descripcion}"]

    categorias = entrada.get("categorias", {})
    if categorias:
        # La codificacion ordinal se deriva de "codificacion": "ordinal ['Bad', ...]"
        orden = re.findall(r"['\"]([^'\"]+)['\"]", str(entrada.get("codificacion", "")))
        codigos = (
            ", ".join(f"{valor}={indice}" for indice, valor in enumerate(orden))
            if orden
            else ""
        )
        permitidos = ", ".join(orden) or ", ".join(categorias.keys())
        partes.append(f"valores permitidos: {permitidos}")
        if codigos:
            partes.append(f"codificacion ordinal: {codigos}")
    else:
        partes.append(
            f"rango tipico del dataset: minimo {entrada.get('min')}, "
            f"media {entrada.get('media')}, maximo {entrada.get('max')}"
        )
    if entrada.get("unidad"):
        partes.append(f"unidad: {entrada['unidad']}")
    return " | ".join(partes)


def construir_extraction_prompt(
    texto_usuario: str, metadata: dict[str, Any] | None = None
) -> str:
    """Rellena EXTRACTION_PROMPT con el texto del usuario y las 10 variables.

    Se usa .replace() en lugar de .format() porque la plantilla contiene un
    ejemplo con llaves JSON y los datos del usuario pueden traer llaves tambien.
    """
    metadata = metadata or cargar_metadata()
    nombres = metadata.get("caracteristicas_ordenadas", list(TIPOS))
    descripciones = "\n".join(f"{i}. {_descriptor(n, metadata)}" for i, n in enumerate(nombres, 1))
    return (
        EXTRACTION_PROMPT.replace("{caracteristicas}", descripciones).replace(
            "{texto_usuario}", texto_usuario
        )
    )


# Etiqueta oficial que debe abrir la justificacion, segun el nivel de la API.
ETIQUETAS_RIESGO = {
    "low": "Riesgo bajo",
    "medium": "Riesgo medio",
    "high": "Riesgo alto",
}


def _formato_factores(top_factors: list[dict[str, Any]] | None) -> str:
    """Traduce los factores de la API (valor real + peso %) a frases legibles."""
    if not top_factors:
        return "No disponible."
    lineas = []
    for factor in top_factors:
        lineas.append(
            f"- {factor.get('feature')}: {factor.get('value')} "
            f"(peso {factor.get('weight_pct')}% del modelo)"
        )
    return "\n".join(lineas)


def construir_justificacion_prompt(
    prediccion: dict[str, Any],
    regla: str,
) -> str:
    """Rellena JUSTIFICATION_PROMPT con la etiqueta oficial, factores y regla."""
    riesgo = str(prediccion.get("risk_level", ""))
    etiqueta = ETIQUETAS_RIESGO.get(riesgo, riesgo or "Riesgo medio")
    reemplazos = {
        "{etiqueta}": etiqueta,
        "{regla}": regla,
        "{factores}": _formato_factores(prediccion.get("top_factors")),
    }
    prompt = JUSTIFICATION_PROMPT
    for marcador, valor in reemplazos.items():
        prompt = prompt.replace(marcador, str(valor))
    return prompt
