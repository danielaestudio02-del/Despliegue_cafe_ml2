# Café ML2 — Segmentación de transacciones de café

## Objetivo

Agrupar las transacciones de ventas de café en **tipos de compra** usando aprendizaje **no supervisado** (Ruta B: K-Means), y desplegar el modelo en una aplicación Streamlit empaquetada con Docker en una VM de Google Cloud.

## Dataset

- **Fuente:** ventas de café de dos tiendas, obtenidas de Kaggle: https://www.kaggle.com/datasets/maurosteban99/datos-caf-limpios
- **Archivo:** `clean_cafe_saless.csv` (3.477 filas, 8 columnas, sin valores faltantes).
- **Unidad de observación:** cada fila es **una transacción** (no un cliente). Por eso los clusters se interpretan como *tipos de compra*.

| Columna | Tipo | ¿Se usa en el modelo? |
|---|---|---|
| `Transaction ID` | Identificador | No (es un identificador) |
| `Item` | Categórica (8 productos) | No (versión base solo con numéricas) |
| `Quantity` | Numérica (1 a 5) | **Sí** |
| `Price Per Unit` | Numérica (1 a 5) | **Sí** |
| `Total Spent` | Numérica (1 a 20) | **Sí** |
| `Payment Method` | Categórica (3) | No |
| `Location` | Categórica (In-store / Takeaway) | No |
| `Transaction Date` | Fecha | No (no tiene sentido como entrada de un caso nuevo) |

### Limitaciones de los datos

1. **`Total Spent` es casi redundante:** en teoría es `Quantity × Price Per Unit`, y así se cumple en el 89% de las filas. Se mantiene para seguir la guía, pero hace que el gasto total pese más al formar los grupos.
2. **Posible imputación previa:** cada producto tiene un precio fijo (Salad 5, Sandwich 4, Coffee 2…), pero unas 137 filas tienen precio 3.0 en productos que no cuestan eso. Además, en 388 filas (11%) `Total Spent` no coincide con cantidad × precio. Los valores que se repiten (cantidad 3, precio 3, total 8) son justo las medianas, así que probablemente quien limpió la base rellenó faltantes con la mediana. En esta versión base no se corrigen; se dejan documentadas.
3. **Datos muy discretos:** cantidad y precio toman pocos valores, por lo que los clusters dividen una "cuadrícula" más que descubrir grupos naturales.

## Ruta elegida

**Ruta B — Aprendizaje no supervisado** (no hay una variable objetivo; se buscan grupos).

## Preparación de datos y modelo

Todo está dentro de un `Pipeline` de scikit-learn, para que la app aplique exactamente las mismas transformaciones que el entrenamiento:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`
3. `KMeans(n_clusters=3, random_state=42, n_init="auto")`

**Justificación:** K-Means es el modelo recomendado por la guía para despliegue porque permite asignar casos nuevos con `predict()`. `random_state=42` hace que el resultado sea reproducible.

## Evaluación

- **Silhouette:** 0.39 (estructura aceptable; va de -1 a 1).
- **Perfiles promedio por cluster:**

| Cluster | Cantidad | Precio unitario | Gasto total | Transacciones | Interpretación |
|---|---|---|---|---|---|
| 0 | 3.9 | $3.89 | $14.72 | 1.168 | Compra grande de productos caros (ensaladas, sándwiches, smoothies) |
| 1 | 3.8 | $1.84 | $6.58 | 1.015 | Muchas unidades de productos baratos (té, galletas, café) |
| 2 | 1.5 | $2.83 | $4.22 | 1.294 | Compra pequeña / individual |

## Interpretación de la salida

La aplicación recibe `Quantity`, `Price Per Unit` y `Total Spent` de una transacción nueva y devuelve el **cluster asignado** con una descripción del tipo de compra.

### Ejemplo de entrada y salida

| Quantity | Price Per Unit | Total Spent | Cluster | Tipo de compra |
|---|---|---|---|---|
| 1 | 2.0 | 2.0 | 2 | Compra pequeña / individual (1 café) |
| 5 | 1.0 | 5.0 | 1 | Muchas unidades baratas (5 galletas) |
| 4 | 5.0 | 20.0 | 0 | Compra grande de productos caros (4 ensaladas) |

## Estructura del proyecto

```
.
├── README.md
├── train.py                    # Script reproducible de entrenamiento
├── model.joblib                # Modelo entrenado (generado en Colab)
├── clean_cafe_saless.csv       # Dataset (lo lee el notebook desde GitHub)
├── data/
│   └── clean_cafe_saless.csv   # Dataset que usa train.py
└── notebooks/
    └── cafe_ml2_modelo.ipynb   # Parte 1: construcción del modelo
```

`train.py` y `model.joblib` se generaron ejecutando el notebook en Google Colab (carpeta `/content/ml_streamlit_project/`) y se subieron a la raíz del repositorio.

## Cómo ejecutar el notebook

1. Abrir `notebooks/cafe_ml2_modelo.ipynb` en Google Colab.
2. Ejecutar todas las celdas (*Entorno de ejecución → Ejecutar todas*). Los datos se leen directamente desde este repositorio.

## Cómo ejecutar `train.py`

Desde la carpeta del proyecto (donde está `data/clean_cafe_saless.csv`):

```bash
python train.py
```

Esto reconstruye `model.joblib` desde cero. El notebook verifica que el modelo generado por `train.py` asigna los mismos clusters que el entrenado en las celdas.

### Contenido de `model.joblib`

Diccionario con las llaves que usa la app de Streamlit:

- `model`: Pipeline entrenado
- `features`: `["Quantity", "Price Per Unit", "Total Spent"]`
- `numeric_features`: igual que `features`
- `task`: `"clustering"`
- `cluster_descriptions`: descripción de cada cluster

## Reproducibilidad

- `random_state=42` en K-Means.
- Preprocesamiento dentro del Pipeline.
- `train.py` reconstruye el modelo con un solo comando.
- **Versiones de librerías:** la última celda del notebook imprime las versiones exactas usadas en Colab. Deben copiarse en `requirements.txt`, porque un `model.joblib` guardado con una versión de scikit-learn puede fallar al cargarse con otra dentro de Docker.
  - El `model.joblib` actual fue entrenado en Colab con estas versiones, que deben usarse en `requirements.txt`:

    ```
    streamlit==1.66.1
    pandas==2.2.3
    numpy==2.4.6
    scikit-learn==1.6.1
    joblib==1.6.0
    ```

## Despliegue (Parte 2)

_Pendiente: cómo ejecutar Streamlit, construir la imagen Docker, ejecutar el contenedor y desplegar en GCP._

## Autores

_Pendiente: nombres de la pareja._
