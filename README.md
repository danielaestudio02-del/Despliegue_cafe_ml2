# Café ML2 — Segmentación de transacciones de café

## Objetivo

Agrupar las transacciones de ventas de café en **tipos de compra** usando aprendizaje **no supervisado** (Ruta B: K-Means), y desplegar el modelo en una aplicación Streamlit empaquetada con Docker en una VM de Google Cloud Compute Engine.

## Dataset

- **Fuente:** Ventas de café de dos tiendas, obtenidas de Kaggle: https://www.kaggle.com/datasets/maurosteban99/datos-caf-limpios
- **Archivo:** `data/clean_cafe_saless.csv` (3.477 filas, 8 columnas, sin valores faltantes).
- **Unidad de observación:** Cada fila es **una transacción** individual (no un cliente). Por eso los clusters se interpretan como *tipos de compra*.

| Columna | Tipo | ¿Se usa en el modelo? |
|---|---|---|
| `Transaction ID` | Identificador | No (es un identificador sin valor predictivo) |
| `Item` | Categórica (8 productos) | No (versión base enfocada en magnitudes numéricas) |
| `Quantity` | Numérica (1 a 5) | **Sí** |
| `Price Per Unit` | Numérica (1 a 5) | **Sí** |
| `Total Spent` | Numérica (1 a 20) | **Sí** |
| `Payment Method` | Categórica (3) | No |
| `Location` | Categórica (In-store / Takeaway) | No |
| `Transaction Date` | Fecha | No (no tiene sentido como entrada en un caso nuevo) |

### Limitaciones de los datos

1. **`Total Spent` es casi redundante:** En teoría es `Quantity × Price Per Unit`, y así se cumple en el 89% de las filas. Se mantiene según la guía, pero hace que el gasto total pese más al formar los grupos.
2. **Posible imputación previa:** Cada producto tiene un precio estándar aproximado, pero unas 137 filas tienen precio 3.0 en productos que no corresponden. Además, en 388 filas (11%) `Total Spent` difiere de cantidad × precio debido a medianas imputadas previamente.
3. **Datos discretos:** Cantidad y precio toman rangos acotados (1 a 5), por lo que los clusters reflejan segmentos operativos bien definidos en el consumo.

## Ruta elegida

**Ruta B — Aprendizaje no supervisado** (segmentación de transacciones con K-Means).

## Preparación de datos y modelo

Todo el flujo está encapsulado en un `Pipeline` de scikit-learn para garantizar total reproducibilidad tanto en entrenamiento como en inferencia dentro de Streamlit:

1. `SimpleImputer(strategy="median")`: Maneja posibles faltantes numéricos.
2. `StandardScaler()`: Estandariza las escalas para que magnitudes de gasto no distorsionen el cálculo de distancias euclidianas.
3. `KMeans(n_clusters=3, random_state=42, n_init="auto")`: Segmenta las transacciones en 3 grupos principales.

**Justificación:** K-Means es ideal para despliegue productivo ligero porque asigna observaciones nuevas mediante el método `.predict()`. Fijar `random_state=42` garantiza resultados 100% reproducibles.

## Evaluación

- **Silhouette Score:** ~0.39 (indica una separación y cohesión estructural adecuada).
- **Perfiles promedio por cluster:**

| Cluster | Cantidad media | Precio unitario medio | Gasto total medio | Transacciones | Interpretación comercial |
|---|---|---|---|---|---|
| **0** | 3.9 uds | $3.89 | $14.72 | 1.168 | **Compra grande / tickets altos** (ensaladas, smoothies, sándwiches) |
| **1** | 3.8 uds | $1.84 | $6.58 | 1.015 | **Volumen de productos económicos** (galletas, té, café en cantidad) |
| **2** | 1.5 uds | $2.83 | $4.22 | 1.294 | **Compra individual / de paso** (1 café o snack rápido) |

## Interpretación de la salida

La aplicación Streamlit recibe `Quantity`, `Price Per Unit` y `Total Spent` de una nueva transacción y calcula en tiempo real el cluster asignado, mostrando la descripción de la compra y su perfil sugerido.

### Ejemplo de entrada y salida

| Caso de prueba | Quantity | Price Per Unit | Total Spent | Cluster Esperado | Tipo de compra |
|---|---|---|---|---|---|
| **Caso 1** | 1.0 | 2.0 | 2.0 | **Cluster 2** | Compra individual / ticket bajo (1 café) |
| **Caso 2** | 5.0 | 1.0 | 5.0 | **Cluster 1** | Muchas unidades de bajo costo (5 galletas) |
| **Caso 3** | 4.0 | 5.0 | 20.0 | **Cluster 0** | Compra grande de productos caros (4 ensaladas) |

---

## Estructura del proyecto

```text
ml_streamlit_project/
├── app.py                      # Aplicación interactiva Streamlit
├── train.py                    # Script reproducible de entrenamiento
├── model.joblib                # Modelo empaquetado y serializado
├── requirements.txt            # Dependencias del proyecto
├── Dockerfile                  # Instrucciones para la imagen Docker
├── .dockerignore               # Archivos omitidos en la construcción Docker
├── README.md                   # Documentación completa y guía de despliegue
├── data/
│   └── clean_cafe_saless.csv   # Dataset de ventas
└── notebooks/
    └── cafe_ml2_modelo.ipynb   # Cuaderno de análisis exploratorio previo
```

---

## Instrucciones de Reproducción y Despliegue

### 1. Reconstruir el modelo localmente (`train.py`)
Para reentrenar el modelo y generar nuevamente el artefacto `model.joblib`:
```bash
python train.py
```

### 2. Ejecutar la aplicación Streamlit localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
Acceder en el navegador a `http://localhost:8501`.

### 3. Construir la imagen Docker
Desde la raíz del repositorio:
```bash
docker build -t ml-streamlit-app .
```

### 4. Ejecutar el contenedor Docker
- **Prueba interactiva:**
  ```bash
  docker run --rm -p 8501:8501 ml-streamlit-app
  ```
- **Ejecución en segundo plano (daemon):**
  ```bash
  docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
  ```
- **Monitoreo y logs:**
  ```bash
  docker ps
  docker logs ml-streamlit
  ```

---

## Despliegue en Google Cloud Compute Engine (GCP)

1. **Crear Instancia VM en GCP:**
   - Sistema Operativo: Ubuntu 24.04 LTS (o Ubuntu 22.04 LTS).
   - Tipo de máquina: `e2-micro` o `e2-small` (económicas y suficientes).
   - Asegurar que tenga asignada una **IP externa efímera o estática**.
   - Asignar una etiqueta de red (Network Tag): `streamlit-server`.

2. **Configurar regla de firewall en GCP:**
   - Crear una regla de ingreso para permitir el puerto 8501:
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --allow=tcp:8501 \
       --direction=INGRESS \
       --target-tags=streamlit-server
   ```
   *(También puede crearse desde la consola web en Red de VPC > Cortafuegos).*

3. **Conectarse vía SSH a la VM e instalar dependencias:**
   ```bash
   sudo apt update
   sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
   *(Cerrar sesión SSH y volver a ingresar para aplicar el grupo docker).*

4. **Clonar el proyecto y desplegar:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO_GITHUB>
   cd Despliegue_cafe_ml2
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```

5. **Acceso público:**
   Ingresar desde cualquier navegador a:
   `http://<IP_EXTERNA_DE_TU_VM>:8501`

---

## Autores
- Integrante 1: *(Nombre completo y código)*
- Integrante 2: *(Nombre completo y código)*
