# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import joblib
import gzip
import os
import zipfile
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import json

def load_and_clean_data(file_path):
    """
    Carga y limpia un DataFrame desde un archivo ZIP que contiene un CSV.

    Args:
        file_path (str): Ruta del archivo ZIP.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        extracted_file = zip_ref.namelist()[0]
        with zip_ref.open(extracted_file) as f:
            df = pd.read_csv(f)

    # Renombrar columna objetivo
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminar columna 'ID'
    if 'ID' in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Eliminar registros con información no disponible
    df.dropna(inplace=True)

    # Agrupar valores > 4 de 'EDUCATION' en "others"
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return df

def save_model(model, output_path):
    """
    Guarda un modelo entrenado en formato comprimido .pkl.gz.

    Args:
        model: Modelo entrenado a guardar.
        output_path (str): Ruta completa donde se guardará el modelo, incluyendo el nombre del archivo.

    Returns:
        None
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, 'wb') as f:
        joblib.dump(model, f)

    print(f"Modelo guardado correctamente en: {output_path}")

def calculate_and_save_metrics(model, X_train, y_train, X_test, y_test, output_path):
    """
    Calcula métricas para conjuntos de entrenamiento y prueba, y las guarda en un archivo JSON en el formato especificado.
    """
    def calculate_metrics(X, y, dataset_name):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])  # Aseguramos etiquetas 0 y 1
        metrics = {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": precision_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }
        cm_matrix = {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }
        return metrics, cm_matrix

    train_metrics, train_cm = calculate_metrics(X_train, y_train, "train")
    test_metrics, test_cm = calculate_metrics(X_test, y_test, "test")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar métricas línea por línea
    with open(output_path, "w", encoding="utf-8") as f:
        for result in [train_metrics, test_metrics, train_cm, test_cm]:
            f.write(json.dumps(result) + "\n")

    print(f"Métricas guardadas correctamente en: {output_path}")


if __name__ == "__main__":
    train_file_path = "files/input/train_data.csv.zip"
    test_file_path = "files/input/test_data.csv.zip"

    print("Cargando y limpiando los datos...")
    train_df = load_and_clean_data(train_file_path)
    test_df = load_and_clean_data(test_file_path)

    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    categorical_features = [col for col in ['SEX', 'EDUCATION', 'MARRIAGE'] if col in X_train.columns]
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10]
    }

    print("Realizando búsqueda de hiperparámetros...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    model_path = "files/models/model.pkl.gz"
    save_model(grid_search, model_path)

    metrics_path = "files/output/metrics.json"
    calculate_and_save_metrics(grid_search, X_train, y_train, X_test, y_test, metrics_path)

    print("Proceso completado.")
