"""Café ML2 · Aplicación Streamlit para clasificar transacciones de café.

Carga el artefacto entrenado (model.joblib) y asigna cada transacción nueva a
uno de los 3 tipos de compra encontrados con K-Means. No vuelve a entrenar.
"""

import base64
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model.joblib")
DATA_PATH = Path("data/clean_cafe_saless.csv")
FONDO_PATH = Path("assets/cafeteria.svg")

# Nombres y textos de cada cluster (salen de la tabla de perfiles del notebook).
CLUSTER_INFO = {
    0: {
        "nombre": "El Gran Antojo",
        "icono": "🥪",
        "color": "#2E7D5B",
        "resumen": "Varias unidades de lo más premium: ensaladas, sándwiches y smoothies.",
        "recomendacion": "Ofrézcale un combo premium o puntos de fidelidad.",
    },
    1: {
        "nombre": "Para Compartir",
        "icono": "🍪",
        "color": "#C98A2B",
        "resumen": "Muchas unidades a buen precio: café, té y galletas.",
        "recomendacion": "Ideal para promociones por cantidad, como un 4x3.",
    },
    2: {
        "nombre": "Café al Paso",
        "icono": "☕",
        "color": "#4A6FA5",
        "resumen": "Una o dos unidades, rápido y sencillo.",
        "recomendacion": "Sugiera un acompañamiento al momento de pagar.",
    },
}

EJEMPLOS = {
    "☕ Café": (1, 2.0),
    "🍪 Galletas": (5, 1.0),
    "🥗 Ensaladas": (4, 5.0),
}

st.set_page_config(
    page_title="Café ML2 · Segmentación de compras",
    page_icon="☕",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 3.5rem; max-width: 1200px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 14px;
             background: linear-gradient(120deg, #3B2A20 0%, #6F4E37 60%, #A47551 100%);
             color: #FFF8F0; margin-bottom: 1rem;}
      .hero h1 {color: #FFF8F0; margin: 0 0 .3rem 0; font-size: 2.1rem;}
      .hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
      .chip {display: inline-block; padding: .15rem .6rem; margin: .5rem .35rem 0 0;
             border-radius: 999px; background: rgba(255,255,255,.16); font-size: .82rem;}
      .result {border-radius: 14px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: .8rem;}
      .result h2 {color: #fff; margin: 0; font-size: 1.6rem;}
      .result p {margin: .3rem 0 0 0; opacity: .95;}
      .footer {text-align: center; opacity: .65; font-size: .85rem; padding: 1.5rem 0 .5rem 0;}
      [data-testid="stSidebar"] {background: linear-gradient(180deg, #3B2618 0%, #2A1A10 100%);}
      [data-testid="stSidebar"] * {color: #F4EBDD;}
      [data-testid="stSidebar"] hr {border-color: rgba(244,235,221,.25);}
      [data-testid="stSidebar"] [data-testid="stExpander"] details {
        background: rgba(255,255,255,.06); border-color: rgba(244,235,221,.2);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def fondo_base64() -> str:
    if not FONDO_PATH.exists():
        return ""
    return base64.b64encode(FONDO_PATH.read_bytes()).decode()


_fondo = fondo_base64()
if _fondo:
    _url = f"url('data:image/svg+xml;base64,{_fondo}')"
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: linear-gradient(rgba(43,27,17,.05), rgba(43,27,17,.15)),
                        {_url} center / cover fixed;
          }}
          [data-testid="stHeader"] {{background: transparent;}}
          [data-testid="stMain"] .block-container {{
            background: rgba(255,250,244,.80);
            backdrop-filter: blur(3px);
            border-radius: 18px;
            padding: 1.6rem 2.2rem 1rem 2.2rem;
            margin-top: 3rem;
            max-width: 1080px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(30,18,10,.35);
          }}
          .hero {{
            background: linear-gradient(90deg, rgba(43,27,17,.90) 0%, rgba(43,27,17,.62) 55%, rgba(43,27,17,.25) 100%),
                        {_url} center / cover;
            padding: 2.2rem 1.8rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- Carga
@st.cache_resource
def cargar_artefacto():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_datos():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


try:
    artifact = cargar_artefacto()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar {MODEL_PATH}: {e}")
    st.stop()

model = artifact["model"]
features = artifact["features"]
df = cargar_datos()

if df is not None:
    df = df.copy()
    df["cluster"] = model.predict(df[features])

if "historial" not in st.session_state:
    st.session_state.historial = []
if "cantidad" not in st.session_state:
    st.session_state.cantidad = 2
    st.session_state.precio = 3.0


def clasificar(cantidad: float, precio: float, total: float):
    x = pd.DataFrame([[cantidad, precio, total]], columns=features)
    cluster = int(model.predict(x)[0])
    # Distancia (en la escala estandarizada) a cada centro de cluster.
    distancias = model.transform(x)[0]
    return cluster, distancias


def nombre_cluster(c: int) -> str:
    info = CLUSTER_INFO.get(c, {})
    return f"{info.get('icono', '')} {info.get('nombre', '')}"


# ---------------------------------------------------------------- Barra lateral
with st.sidebar:
    st.markdown("## ☕ Café ML2")
    st.caption("Segmentación de compras en una cafetería.")
    st.divider()
    st.markdown("### Tipos de compra")
    for c, info in CLUSTER_INFO.items():
        st.markdown(f"{info['icono']} **{info['nombre']}**")
    st.divider()
    with st.expander("Sobre el modelo"):
        st.markdown(
            "- **K-Means** con 3 grupos\n"
            "- Imputación (mediana) + estandarización\n"
            "- Variables: cantidad, precio unitario y gasto total\n"
            "- Datos: ventas de dos cafeterías (Kaggle); cada fila es una transacción"
        )
    st.caption("Streamlit + Docker · Ruta B: K-Means")


# ---------------------------------------------------------------- Encabezado
st.markdown(
    """
    <div class="hero">
      <h1>☕ Segmentación de Ventas de Café</h1>
      <p>Ingrese una transacción y descubra a qué <b>tipo de compra</b> pertenece,
      según un modelo K-Means entrenado con ventas reales de dos cafeterías.</p>
      <span class="chip">🤖 K-Means · 3 tipos de compra</span>
      <span class="chip">🐳 Desplegado con Docker</span>
      <span class="chip">📦 Modelo pre-entrenado (model.joblib)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Transacciones analizadas", f"{len(df):,}".replace(",", ".") if df is not None else "3.477")
m2.metric("Tipos de compra", "3")
m3.metric("Variables del modelo", str(len(features)))

tab_clasificar, tab_lote, tab_perfiles, tab_datos, tab_como = st.tabs(
    ["🔍 Clasificar", "📁 Clasificar archivo", "📊 Perfiles", "🗂️ Datos", "ℹ️ Cómo funciona"]
)

# ---------------------------------------------------------------- Clasificar
with tab_clasificar:
    col_form, col_res = st.columns([1, 1.25], gap="large")

    with col_form:
        st.subheader("📋 Datos de la transacción")
        st.caption("Pruebe un ejemplo rápido o ingrese sus propios valores.")
        e_cols = st.columns(len(EJEMPLOS))
        for col, (etiqueta, (q, p)) in zip(e_cols, EJEMPLOS.items()):
            if col.button(etiqueta, width="stretch"):
                st.session_state.cantidad = q
                st.session_state.precio = p

        with st.container(border=True):
            cantidad = st.number_input(
                "Cantidad de unidades (Quantity)",
                min_value=1, max_value=20, step=1, key="cantidad",
                help="En los datos de entrenamiento va de 1 a 5 unidades.",
            )
            precio = st.number_input(
                "Precio unitario (Price Per Unit) [$]",
                min_value=0.5, max_value=20.0, step=0.5, key="precio",
                help="En los datos de entrenamiento va de $1 a $5.",
            )
            manual = st.toggle(
                "Ingresar el gasto total manualmente",
                help="Por defecto se calcula como cantidad × precio unitario.",
            )
            if manual:
                total = st.number_input(
                    "Gasto total (Total Spent) [$]",
                    min_value=0.5, max_value=400.0, step=0.5,
                    value=float(cantidad * precio),
                )
            else:
                total = float(cantidad * precio)
                st.metric("Gasto total (Total Spent)", f"${total:,.2f}")

            if cantidad > 5 or precio > 5:
                st.warning(
                    "Algunos valores están fuera del rango de los datos de entrenamiento "
                    "(1 a 5). El modelo igual asigna un cluster, pero con menos respaldo."
                )

            clasificar_btn = st.button("🔍 Clasificar tipo de compra", type="primary", width="stretch")

    with col_res:
        st.subheader("🎯 Resultado")
        if clasificar_btn:
            cluster, distancias = clasificar(cantidad, precio, total)
            st.session_state.ultimo = (cantidad, precio, total, cluster, distancias)
            st.session_state.historial.insert(0, {
                "Hora": datetime.now().strftime("%H:%M:%S"),
                "Quantity": cantidad,
                "Price Per Unit": precio,
                "Total Spent": total,
                "Tipo de compra": CLUSTER_INFO[cluster]["nombre"],
            })

        if "ultimo" not in st.session_state:
            st.info("Complete los datos y presione **Clasificar tipo de compra** para ver el resultado.")
        else:
            cantidad_r, precio_r, total_r, cluster, distancias = st.session_state.ultimo
            info = CLUSTER_INFO[cluster]
            st.markdown(
                f"""
                <div class="result" style="background:{info['color']}">
                  <p style="margin:0;opacity:.85">Tipo de compra</p>
                  <h2>{info['icono']} {info['nombre']}</h2>
                  <p>{info['resumen']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**💡 Recomendación:** {info['recomendacion']}")

            # Cercanía a cada tipo de compra (inverso de la distancia al centro).
            cercania = 1 / (distancias + 1e-9)
            cercania = cercania / cercania.sum()
            dist_df = pd.DataFrame({
                "Tipo de compra": [nombre_cluster(c) for c in range(len(distancias))],
                "Cercanía (%)": np.round(cercania * 100, 1),
                "color": [CLUSTER_INFO[c]["color"] for c in range(len(distancias))],
            })
            st.markdown("**Cercanía a cada tipo de compra**")
            chart = (
                alt.Chart(dist_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Cercanía (%):Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Tipo de compra:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["Tipo de compra", "Cercanía (%)"],
                )
                .properties(height=150)
            )
            st.altair_chart(chart, width="stretch")
            st.caption(
                "La cercanía compara la distancia de la transacción al centro de cada cluster "
                "(variables estandarizadas). Es una guía visual, no una probabilidad."
            )

            if df is not None:
                prom = df[df["cluster"] == cluster][features].mean()
                comp = pd.DataFrame({
                    "Variable": features,
                    "Su transacción": [cantidad_r, precio_r, total_r],
                    "Promedio del grupo": prom.values.round(2),
                })
                st.markdown("**Su transacción frente al promedio del grupo**")
                st.dataframe(comp, hide_index=True, width="stretch")

    if st.session_state.historial:
        st.divider()
        h1, h2 = st.columns([3, 1])
        h1.subheader("🕘 Historial de esta sesión")
        hist_df = pd.DataFrame(st.session_state.historial)
        h2.download_button(
            "⬇️ Descargar CSV", hist_df.to_csv(index=False).encode("utf-8"),
            file_name="historial_clasificaciones.csv", mime="text/csv", width="stretch",
        )
        st.dataframe(hist_df, hide_index=True, width="stretch")
        if st.button("🗑️ Borrar historial"):
            st.session_state.historial = []
            st.rerun()

# ---------------------------------------------------------------- Clasificar archivo
with tab_lote:
    st.subheader("📁 Clasificar varias transacciones")
    st.markdown(
        "Suba un archivo **CSV** con las columnas `Quantity`, `Price Per Unit` y `Total Spent`. "
        "La aplicación asigna un tipo de compra a cada fila y permite descargar el resultado."
    )
    plantilla = pd.DataFrame(
        [[1, 2.0, 2.0], [5, 1.0, 5.0], [4, 5.0, 20.0]], columns=features
    )
    st.download_button(
        "⬇️ Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_transacciones.csv", mime="text/csv",
    )
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
            faltan = [c for c in features if c not in lote.columns]
            if faltan:
                st.error(f"Faltan columnas en el archivo: {', '.join(faltan)}")
            else:
                lote["Tipo de compra"] = [nombre_cluster(int(c)) for c in model.predict(lote[features])]
                st.success(f"Se clasificaron {len(lote)} transacciones.")
                resumen = lote["Tipo de compra"].value_counts().rename_axis("Tipo de compra").reset_index(name="Transacciones")
                c1, c2 = st.columns([1, 2])
                c1.dataframe(resumen, hide_index=True, width="stretch")
                c2.dataframe(lote, hide_index=True, width="stretch", height=260)
                st.download_button(
                    "⬇️ Descargar resultado", lote.to_csv(index=False).encode("utf-8"),
                    file_name="transacciones_clasificadas.csv", mime="text/csv",
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")

# ---------------------------------------------------------------- Perfiles
with tab_perfiles:
    st.subheader("📊 Perfiles de los 3 tipos de compra")
    cols = st.columns(3)
    for col, (c, info) in zip(cols, CLUSTER_INFO.items()):
        with col.container(border=True):
            st.markdown(f"### {info['icono']} {info['nombre']}")
            st.caption(info["resumen"])
            if df is not None:
                sub = df[df["cluster"] == c]
                st.metric("Transacciones", f"{len(sub):,}".replace(",", "."))
                st.caption(f"{len(sub) / len(df):.0%} del total de transacciones")
                st.markdown(
                    f"- Cantidad promedio: **{sub['Quantity'].mean():.1f}** uds\n"
                    f"- Precio unitario promedio: **\\${sub['Price Per Unit'].mean():.2f}**\n"
                    f"- Gasto promedio: **\\${sub['Total Spent'].mean():.2f}**"
                )

    if df is not None:
        st.markdown("#### ¿Cómo se ven los grupos?")
        st.caption("Cada punto es una combinación de cantidad y precio; el tamaño indica cuántas transacciones hay.")
        puntos = (
            df.groupby(["Quantity", "Price Per Unit", "cluster"]).size().reset_index(name="Transacciones")
        )
        puntos["Tipo de compra"] = puntos["cluster"].map(nombre_cluster)
        dominio = [nombre_cluster(c) for c in CLUSTER_INFO]
        colores = [CLUSTER_INFO[c]["color"] for c in CLUSTER_INFO]
        scatter = (
            alt.Chart(puntos)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X("Quantity:Q", title="Cantidad (Quantity)", scale=alt.Scale(domain=[0, 6]), axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("Price Per Unit:Q", title="Precio unitario (Price Per Unit)", scale=alt.Scale(domain=[0, 6]), axis=alt.Axis(tickMinStep=1)),
                size=alt.Size("Transacciones:Q", legend=None),
                color=alt.Color("Tipo de compra:N", scale=alt.Scale(domain=dominio, range=colores),
                                legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
                tooltip=["Tipo de compra", "Quantity", "Price Per Unit", "Transacciones"],
            )
            .properties(height=380)
        )
        st.altair_chart(scatter, width="stretch")

        st.markdown("#### Productos más vendidos en cada tipo de compra")
        prod = df.groupby(["cluster", "Item"]).size().reset_index(name="Transacciones")
        prod["Tipo de compra"] = prod["cluster"].map(nombre_cluster)
        barras = (
            alt.Chart(prod)
            .mark_bar()
            .encode(
                x=alt.X("Transacciones:Q"),
                y=alt.Y("Item:N", title=None, sort="-x", axis=alt.Axis(labelOverlap=False)),
                color=alt.Color("Tipo de compra:N", scale=alt.Scale(domain=dominio, range=colores),
                                legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
                tooltip=["Tipo de compra", "Item", "Transacciones"],
            )
            .properties(height=340)
        )
        st.altair_chart(barras, width="stretch")
    else:
        st.info("No se encontró el archivo de datos; se muestran solo las descripciones del modelo.")

# ---------------------------------------------------------------- Datos
with tab_datos:
    st.subheader("🗂️ Datos de entrenamiento")
    if df is not None:
        st.markdown(
            "Ventas de café de dos tiendas, obtenidas de "
            "[Kaggle](https://www.kaggle.com/datasets/maurosteban99/datos-caf-limpios). "
            "Cada fila es una transacción."
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Filas", f"{len(df):,}".replace(",", "."))
        d2.metric("Productos distintos", df["Item"].nunique())
        d3.metric("Gasto promedio", f"${df['Total Spent'].mean():.2f}")
        st.dataframe(df.drop(columns=["cluster"]).head(200), hide_index=True, width="stretch", height=320)
        st.caption("Se muestran las primeras 200 filas.")
        with st.expander("⚠️ Limitaciones de los datos"):
            st.markdown(
                "- `Total Spent` debería ser cantidad × precio, pero no coincide en ~11% de las filas.\n"
                "- Algunas filas tienen precio 3.0 en productos que no cuestan eso; probablemente "
                "son valores faltantes rellenados con la mediana.\n"
                "- Cantidad y precio toman pocos valores (1 a 5), por lo que los grupos dividen una "
                "\"cuadrícula\" más que descubrir grupos naturales.\n"
                "- Cada fila es una transacción, no un cliente: los clusters son tipos de compra."
            )
    else:
        st.info("El archivo de datos no está disponible en este despliegue.")

# ---------------------------------------------------------------- Cómo funciona
with tab_como:
    st.subheader("ℹ️ ¿Cómo funciona?")
    st.markdown(
        """
        1. **Entrenamiento (notebook / `train.py`):** con las variables `Quantity`,
           `Price Per Unit` y `Total Spent` se entrena un *Pipeline* de scikit-learn:
           imputación por mediana → estandarización → **K-Means con 3 clusters**
           (`random_state=42` para que sea reproducible).
        2. **Artefacto:** el modelo entrenado se guarda en `model.joblib` junto con las
           variables y la descripción de cada cluster.
        3. **Esta aplicación:** carga `model.joblib` **sin volver a entrenar**, aplica las
           mismas transformaciones del entrenamiento y asigna el cluster más cercano.
        4. **Despliegue:** la app se empaqueta con **Docker** (`Dockerfile` →
           `docker build` → imagen → `docker run` → contenedor).
        """
    )
    st.markdown("#### Interpretación")
    st.markdown(
        "Cada tipo de compra se describe con los promedios de su grupo (pestaña **Perfiles**)."
    )
    st.markdown("#### Reproducibilidad")
    st.code(
        "git clone https://github.com/danielaestudio02-del/Despliegue_cafe_ml2.git\n"
        "cd Despliegue_cafe_ml2\n"
        "docker build -t ml-streamlit-app .\n"
        "docker run -p 8501:8501 ml-streamlit-app",
        language="bash",
    )

st.markdown(
    '<div class="footer">Proyecto Reto ML · Streamlit + Docker · Ruta B: K-Means Clustering<br>'
    "Datos: Kaggle · Ventas de café de dos tiendas</div>",
    unsafe_allow_html=True,
)
