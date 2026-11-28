import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TARGET = "Credit_Score"
MAPEO_OBJETIVO = {"Good": 0, "Standard": 1, "Poor": 2}
ETIQUETAS = {0: "Riesgo bajo (Good)", 1: "Riesgo medio (Standard)", 2: "Riesgo alto (Poor)"}

FEATURES = [
    "Credit_Utilization_Ratio",
    "Outstanding_Debt",
    "Interest_Rate",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Credit_History_Age",
    "Num_of_Delayed_Payment",
    "Monthly_Inhand_Salary",
]

NUMERICAS = [
    "Credit_Utilization_Ratio",
    "Outstanding_Debt",
    "Interest_Rate",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Credit_History_Age",
    "Num_of_Delayed_Payment",
    "Monthly_Inhand_Salary",
]

CATEGORICAS = ["Credit_Mix", "Payment_of_Min_Amount"]

CATEGORIAS_ORDINALES = {
    "Credit_Mix": ["Bad", "Standard", "Good"],
    "Payment_of_Min_Amount": ["No", "NM", "Yes"],
}

DESCRIPCIONES = {
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


def cargar_dataset():
    candidatos = sorted(DATA_DIR.glob("*.csv"))
    for ruta in candidatos:
        columnas = pd.read_csv(ruta, nrows=0).columns
        if TARGET in columnas:
            df = pd.read_csv(ruta, low_memory=False)
            print(f"Dataset cargado: {ruta.name} | shape={df.shape}")
            print(f"Columnas detectadas ({len(df.columns)}): {list(df.columns)}")
            return df, ruta.name
    raise FileNotFoundError(
        f"Ningun CSV en {DATA_DIR} contiene la columna objetivo '{TARGET}'. "
        "Se requiere un train.csv etiquetado."
    )


def a_numerico(serie):
    limpia = serie.astype(str).str.replace("_", "", regex=False).str.strip()
    limpia = limpia.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan, "__": np.nan})
    return pd.to_numeric(limpia, errors="coerce")


def a_meses(serie):
    def convertir(valor):
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return np.nan
        if isinstance(valor, (int, float, np.integer, np.floating)):
            return float(valor)
        texto = str(valor).strip()
        if texto in {"", "nan", "None", "_", "__"}:
            return np.nan
        coincidencia = re.match(r"^(\d+)\s*Years?\s*(?:and\s*)?(\d+)\s*Months?$", texto, re.IGNORECASE)
        if coincidencia:
            return int(coincidencia.group(1)) * 12 + int(coincidencia.group(2))
        coincidencia = re.match(r"^(\d+)\s*Months?$", texto, re.IGNORECASE)
        if coincidencia:
            return float(coincidencia.group(1))
        return np.nan

    return serie.apply(convertir)


def limpiar(df):
    reporte = {}
    df = df[FEATURES + [TARGET]].copy()

    df["Credit_Utilization_Ratio"] = a_numerico(df["Credit_Utilization_Ratio"])
    df["Outstanding_Debt"] = a_numerico(df["Outstanding_Debt"])
    df["Interest_Rate"] = a_numerico(df["Interest_Rate"])
    df["Num_Credit_Inquiries"] = a_numerico(df["Num_Credit_Inquiries"])
    df["Delay_from_due_date"] = a_numerico(df["Delay_from_due_date"])
    df["Num_of_Delayed_Payment"] = a_numerico(df["Num_of_Delayed_Payment"])
    df["Monthly_Inhand_Salary"] = a_numerico(df["Monthly_Inhand_Salary"])
    df["Credit_History_Age"] = a_meses(df["Credit_History_Age"])

    df["Credit_Mix"] = df["Credit_Mix"].astype(str).str.strip().replace({"_": np.nan, "nan": np.nan, "": np.nan})
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].astype(str).str.strip()

    reporte["nulos_antes"] = {c: int(df[c].isna().sum()) for c in FEATURES if df[c].isna().sum() > 0}

    for columna in NUMERICAS:
        mediana = df[columna].median()
        df[columna] = df[columna].fillna(mediana)

    moda_credit_mix = df["Credit_Mix"].mode(dropna=True).iloc[0]
    df["Credit_Mix"] = df["Credit_Mix"].fillna(moda_credit_mix)
    reporte["imputacion_credit_mix"] = str(moda_credit_mix)

    reporte["nulos_despues"] = int(df[FEATURES].isna().sum().sum())
    reporte["fuera_de_dominio_credit_mix"] = sorted(
        set(df["Credit_Mix"]) - set(CATEGORIAS_ORDINALES["Credit_Mix"])
    )
    reporte["fuera_de_dominio_payment"] = sorted(
        set(df["Payment_of_Min_Amount"]) - set(CATEGORIAS_ORDINALES["Payment_of_Min_Amount"])
    )
    return df, reporte


def codificar(df):
    categorias = [CATEGORIAS_ORDINALES[c] for c in CATEGORICAS]
    encoder = OrdinalEncoder(
        categories=categorias,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )
    codificado = encoder.fit_transform(df[CATEGORICAS])
    for indice, columna in enumerate(CATEGORICAS):
        df[columna] = codificado[:, indice]
    return df, encoder


def mapear_objetivo(df):
    valores = set(df[TARGET].unique())
    no_mapeados = valores - set(MAPEO_OBJETIVO)
    if no_mapeados:
        raise ValueError(f"Valores de {TARGET} fuera del mapeo exigido: {sorted(no_mapeados)}")
    df[TARGET] = df[TARGET].map(MAPEO_OBJETIVO).astype(int)
    return df


def eliminar_outliers(df):
    filas_iniciales = len(df)
    mascara = pd.Series(True, index=df.index)
    limites = {}
    for columna in NUMERICAS:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr
        limites[columna] = {"min": float(inferior), "max": float(superior)}
        mascara &= df[columna].between(inferior, superior)
    df = df[mascara].reset_index(drop=True)
    return df, filas_iniciales - len(df), limites


def evaluar(nombre, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    metricas = {
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 4),
        "f1_macro": round(float(f1_score(y_test, predicciones, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, predicciones, average="weighted")), 4),
        "matriz_confusion": confusion_matrix(y_test, predicciones).tolist(),
    }
    print(f"\n--- {nombre} ---")
    print(f"  Accuracy        : {metricas['accuracy']:.4f}")
    print(f"  F1-Score macro  : {metricas['f1_macro']:.4f}")
    print(f"  F1-Score weighted: {metricas['f1_weighted']:.4f}")
    print("  Matriz de confusion:")
    matriz = np.array(metricas["matriz_confusion"])
    print(f"    {'':>16}" + "".join(f"  pred_{c}" for c in range(3)))
    for i, fila in enumerate(matriz):
        print(f"    real {i} ({ETIQUETAS[i][:14]:>14})" + "".join(f"  {v:6d}" for v in fila))
    return metricas


def calcular_importancias(modelo, X_test, y_test):
    if hasattr(modelo, "feature_importances_"):
        valores = np.asarray(modelo.feature_importances_, dtype=float)
        origen = "feature_importances_"
    elif hasattr(modelo, "coef_"):
        valores = np.mean(np.abs(np.atleast_2d(modelo.coef_)), axis=0)
        origen = "coef_"
    else:
        resultado = permutation_importance(
            modelo, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        valores = resultado.importances_mean
        origen = "permutation_importance"
    total = float(np.sum(np.abs(valores))) or 1.0
    return {n: round(float(v) / total, 4) for n, v in zip(FEATURES, valores)}, origen


def rangos_tipicos(df):
    rangos = {}
    for columna in FEATURES:
        serie = df[columna]
        rangos[columna] = {
            "tipo": "numerica" if columna in NUMERICAS else "categorica_ordinal",
            "min": round(float(serie.min()), 4),
            "media": round(float(serie.mean()), 4),
            "max": round(float(serie.max()), 4),
        }
        if columna in CATEGORICAS:
            conteos = serie.value_counts().sort_index()
            rangos[columna]["categorias"] = {
                CATEGORIAS_ORDINALES[columna][int(clave)]: int(valor)
                for clave, valor in conteos.items()
                if 0 <= int(clave) < len(CATEGORIAS_ORDINALES[columna])
            }
    return rangos


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, nombre_dataset = cargar_dataset()
    print(f"\nObjetivo {TARGET}: {df[TARGET].value_counts().to_dict()}")

    df, reporte_limpieza = limpiar(df)
    print("\n[2] Limpieza")
    print(f"  Nulos detectados antes de imputar: {reporte_limpieza['nulos_antes']}")
    print(f"  Nulos restantes tras imputar     : {reporte_limpieza['nulos_despues']}")
    print(f"  Credit_Mix imputado con la moda  : {reporte_limpieza['imputacion_credit_mix']}")
    print(f"  Valores fuera de dominio (Mix)   : {reporte_limpieza['fuera_de_dominio_credit_mix']}")
    print(f"  Valores fuera de dominio (Payment): {reporte_limpieza['fuera_de_dominio_payment']}")

    df, encoder = codificar(df)
    joblib.dump(encoder, MODELS_DIR / "encoder.joblib")
    print("\n[2b] Codificador ordinal guardado en models/encoder.joblib")

    df = mapear_objetivo(df)
    print(f"\n[3] Objetivo mapeado: {MAPEO_OBJETIVO} -> {ETIQUETAS}")

    df, eliminadas, limites = eliminar_outliers(df)
    print(f"\n[4] Outliers IQR eliminados: {eliminadas} filas | restantes: {len(df)}")

    X = df[FEATURES].astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[5] Split 80/20 con stratify -> train={len(X_train)} test={len(X_test)}")
    print(f"    Distribucion train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"    Distribucion test : {y_test.value_counts().sort_index().to_dict()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("\n[6] StandardScaler ajustado solo sobre X_train")

    modelos = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "SVC(probability=True)": SVC(probability=True, random_state=42),
    }

    print("\n[7] Entrenamiento y comparacion de modelos")
    resultados = {}
    objetos = {}
    for nombre, modelo in modelos.items():
        resultados[nombre] = evaluar(nombre, modelo, X_train, y_train, X_test, y_test)
        objetos[nombre] = modelo

    mejor_nombre = max(resultados, key=lambda n: (resultados[n]["f1_weighted"], resultados[n]["accuracy"]))
    mejor_modelo = objetos[mejor_nombre]
    print(f"\n  Modelo ganador (f1_weighted): {mejor_nombre}")

    importancias, origen_importancia = calcular_importancias(mejor_modelo, X_test, y_test)

    joblib.dump(mejor_modelo, MODELS_DIR / "modelo_riesgo.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print("\n[8] Artefactos guardados:")
    print("    models/modelo_riesgo.joblib")
    print("    models/scaler.joblib")

    rangos = rangos_tipicos(df)
    caracteristicas = []
    for columna in FEATURES:
        entrada = {
            "nombre": columna,
            "orden": FEATURES.index(columna) + 1,
            "tipo": rangos[columna]["tipo"],
            "descripcion": DESCRIPCIONES.get(columna, ""),
            "min": rangos[columna]["min"],
            "media": rangos[columna]["media"],
            "max": rangos[columna]["max"],
            "importancia": importancias[columna],
            "codificacion": (
                "ninguna"
                if columna in NUMERICAS and columna not in CATEGORICAS
                else f"ordinal {CATEGORIAS_ORDINALES[columna]}"
            ),
        }
        if "categorias" in rangos[columna]:
            entrada["categorias"] = rangos[columna]["categorias"]
        if columna == "Credit_History_Age":
            entrada["unidad"] = "meses"
        if columna in limites:
            entrada["limites_iqr"] = limites[columna]
        caracteristicas.append(entrada)

    metadata = {
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": nombre_dataset,
        "n_muestras": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "outliers_eliminados_iqr": int(eliminadas),
        "caracteristicas_ordenadas": FEATURES,
        "caracteristicas": caracteristicas,
        "modelo_ganador": mejor_nombre,
        "origen_importancia": origen_importancia,
        "clases": {str(k): v for k, v in ETIQUETAS.items()},
        "mapeo_objetivo": MAPEO_OBJETIVO,
        "metricas": resultados,
        "artefactos": {
            "modelo": "models/modelo_riesgo.joblib",
            "escalador": "models/scaler.joblib",
            "codificador": "models/encoder.joblib",
        },
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, indent=2, ensure_ascii=False)
    print("    models/metadata.json")

    print("\nImportancia de características (modelo ganador):")
    for columna in sorted(FEATURES, key=lambda c: importancias[c], reverse=True):
        print(f"    {columna:<30} {importancias[columna]:.4f}")


if __name__ == "__main__":
    main()
