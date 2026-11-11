"""Café ML2 · Aplicación Streamlit para clasificar compras en una cafetería.

Carga el artefacto entrenado (model.joblib) y asigna cada compra nueva a uno de
los 3 tipos de compra encontrados con K-Means. No vuelve a entrenar.
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

# Nombres y textos de cada tipo de compra (salen de la tabla de perfiles del notebook).
CLUSTER_INFO = {
    0: {
        "nombre": "El Gran Antojo",
        "icono": "🥪",
        "color": "#2E7D5B",
        "resumen": "Varias unidades de lo más premium: ensaladas, sándwiches y batidos.",
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

# Nombres en español de las variables del modelo (las columnas originales están en inglés).
VAR_ES = {"Quantity": "Cantidad", "Price Per Unit": "Precio unitario", "Total Spent": "Gasto total"}
COLUMNAS_ES = {
    "Transaction ID": "ID de transacción",
    "Item": "Producto",
    "Quantity": "Cantidad",
    "Price Per Unit": "Precio unitario",
    "Total Spent": "Gasto total",
    "Payment Method": "Método de pago",
    "Location": "Lugar",
    "Transaction Date": "Fecha",
}
PRODUCTOS_ES = {
    "Coffee": ("☕", "Café"),
    "Tea": ("🍵", "Té"),
    "Cookie": ("🍪", "Galleta"),
    "Cake": ("🍰", "Pastel"),
    "Juice": ("🧃", "Jugo"),
    "Sandwich": ("🥪", "Sándwich"),
    "Smoothie": ("🥤", "Batido"),
    "Salad": ("🥗", "Ensalada"),
}
PAGO_ES = {"Cash": "Efectivo", "Credit Card": "Tarjeta de crédito", "Digital Wallet": "Billetera digital"}
LUGAR_ES = {"In-store": "En tienda", "Takeaway": "Para llevar"}
# Precio habitual de cada producto en los datos (se recalcula si están los datos).
PRECIOS_BASE = {"Coffee": 2.0, "Tea": 1.5, "Cookie": 1.0, "Cake": 3.0, "Juice": 3.0,
                "Sandwich": 4.0, "Smoothie": 4.0, "Salad": 5.0}

EJEMPLOS = {
    "☕ Café": (1, 2.0),
    "🍪 Galletas": (5, 1.0),
    "🥗 Ensaladas": (4, 5.0),
}

st.set_page_config(
    page_title="Café ML2 · Tipos de compra",
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
      /* Ocultar textos en inglés propios de Streamlit */
      [data-testid="stStatusWidget"] {visibility: hidden;}
      [data-testid="stFileUploaderDropzoneInstructions"] > div > span,
      [data-testid="stFileUploaderDropzoneInstructions"] > div > small {display: none;}
      [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
        content: "Arrastre y suelte aquí su archivo CSV"; display: block;}
      [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
        content: "Máximo 200 MB por archivo · CSV"; display: block; font-size: .8rem; opacity: .7;}
      [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] {display: none;}
      [data-testid="stFileUploaderDropzone"] button [data-has-shortcut]::after {content: "Buscar archivo";}
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
    st.error(f"No se pudo cargar el modelo ({MODEL_PATH}): {e}")
    st.stop()

model = artifact["model"]
features = artifact["features"]
df = cargar_datos()

precios = dict(PRECIOS_BASE)
if df is not None:
    df = df.copy()
    df["cluster"] = model.predict(df[features])
    precios.update(df.groupby("Item")["Price Per Unit"].agg(lambda s: s.mode().iloc[0]).to_dict())

for clave, valor in {"historial": [], "pedido": [], "pedido_res": None, "cantidad": 2, "precio": 3.0}.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor


def a_modelo(cantidad, precio, total) -> pd.DataFrame:
    """Arma la tabla con los nombres de columna que espera el modelo."""
    return pd.DataFrame({"Quantity": cantidad, "Price Per Unit": precio, "Total Spent": total})[features]


def clasificar(cantidad: float, precio: float, total: float):
    x = a_modelo([cantidad], [precio], [total])
    cluster = int(model.predict(x)[0])
    distancias = model.transform(x)[0]  # distancia a cada centro (variables estandarizadas)
    return cluster, distancias


def nombre_cluster(c: int) -> str:
    info = CLUSTER_INFO.get(int(c), {})
    return f"{info.get('icono', '')} {info.get('nombre', '')}"


def nombre_producto(item: str) -> str:
    icono, nombre = PRODUCTOS_ES.get(item, ("", item))
    return f"{icono} {nombre}".strip()


def tarjeta_resultado(cluster: int, titulo: str = "Tipo de compra"):
    info = CLUSTER_INFO[cluster]
    st.markdown(
        f"""
        <div class="result" style="background:{info['color']}">
          <p style="margin:0;opacity:.85">{titulo}</p>
          <h2>{info['icono']} {info['nombre']}</h2>
          <p>{info['resumen']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**💡 Recomendación:** {info['recomendacion']}")


# ---------------------------------------------------------------- Barra lateral
with st.sidebar:
    st.markdown("## ☕ Café ML2")
    st.caption("Tipos de compra en una cafetería.")
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
            "- Datos: ventas de dos cafeterías (Kaggle); cada fila es la compra de un producto"
        )
    st.caption("Streamlit + Docker · Ruta B: agrupamiento con K-Means")


# ---------------------------------------------------------------- Encabezado
st.markdown(
    """
    <div class="hero">
      <h1>☕ Tipos de Compra en la Cafetería</h1>
      <p>Ingrese una compra, o un pedido con varios productos, y descubra a qué
      <b>tipo de compra</b> pertenece, según un modelo entrenado con ventas reales
      de dos cafeterías.</p>
      <span class="chip">🤖 K-Means · 3 tipos de compra</span>
      <span class="chip">🐳 Desplegado con Docker</span>
      <span class="chip">📦 Modelo ya entrenado</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Compras analizadas", f"{len(df):,}".replace(",", ".") if df is not None else "3.477")
m2.metric("Tipos de compra", "3")
m3.metric("Productos en la carta", str(len(precios)))

tab_clasificar, tab_lote, tab_perfiles, tab_datos, tab_como = st.tabs(
    ["🔍 Clasificar", "📁 Clasificar archivo", "📊 Perfiles", "🗂️ Datos", "ℹ️ Cómo funciona"]
)

# ---------------------------------------------------------------- Clasificar
with tab_clasificar:
    modo = st.radio(
        "¿Qué desea clasificar?",
        ["🧾 Un producto", "🛒 Un pedido con varios productos"],
        horizontal=True,
    )
    col_form, col_res = st.columns([1, 1.25], gap="large")

    # ------------------------------ Modo: un producto
    if modo == "🧾 Un producto":
        with col_form:
            st.subheader("📋 Datos de la compra")
            st.caption("Pruebe un ejemplo rápido o ingrese sus propios valores.")
            e_cols = st.columns(len(EJEMPLOS))
            for col, (etiqueta, (q, p)) in zip(e_cols, EJEMPLOS.items()):
                if col.button(etiqueta, width="stretch"):
                    st.session_state.cantidad = q
                    st.session_state.precio = p

            with st.container(border=True):
                cantidad = st.number_input(
                    "Cantidad de unidades",
                    min_value=1, max_value=20, step=1, key="cantidad",
                    help="En los datos de entrenamiento va de 1 a 5 unidades.",
                )
                precio = st.number_input(
                    "Precio unitario ($)",
                    min_value=0.5, max_value=20.0, step=0.5, key="precio",
                    help="En los datos de entrenamiento va de $1 a $5.",
                )
                manual = st.toggle(
                    "Ingresar el gasto total manualmente",
                    help="Por defecto se calcula como cantidad × precio unitario.",
                )
                if manual:
                    total = st.number_input(
                        "Gasto total ($)",
                        min_value=0.5, max_value=400.0, step=0.5,
                        value=float(cantidad * precio),
                    )
                else:
                    total = float(cantidad * precio)
                    st.metric("Gasto total", f"${total:,.2f}")

                if cantidad > 5 or precio > 5:
                    st.warning(
                        "Algunos valores están fuera del rango de los datos de entrenamiento "
                        "(1 a 5). El modelo igual asigna un tipo de compra, pero con menos respaldo."
                    )

                clasificar_btn = st.button("🔍 Clasificar tipo de compra", type="primary", width="stretch")

        with col_res:
            st.subheader("🎯 Resultado")
            if clasificar_btn:
                cluster, distancias = clasificar(cantidad, precio, total)
                st.session_state.ultimo = (cantidad, precio, total, cluster, distancias)
                st.session_state.historial.insert(0, {
                    "Hora": datetime.now().strftime("%H:%M:%S"),
                    "Producto": "—",
                    "Cantidad": cantidad,
                    "Precio unitario": precio,
                    "Gasto total": total,
                    "Tipo de compra": nombre_cluster(cluster),
                })

            if "ultimo" not in st.session_state:
                st.info("Complete los datos y presione **Clasificar tipo de compra** para ver el resultado.")
            else:
                cantidad_r, precio_r, total_r, cluster, distancias = st.session_state.ultimo
                tarjeta_resultado(cluster)

                # Cercanía a cada tipo de compra (inverso de la distancia al centro del grupo).
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
                    "La cercanía compara la distancia de la compra al centro de cada grupo "
                    "(variables estandarizadas). Es una guía visual, no una probabilidad."
                )

                if df is not None:
                    prom = df[df["cluster"] == cluster][features].mean()
                    comp = pd.DataFrame({
                        "Variable": [VAR_ES[f] for f in features],
                        "Su compra": [cantidad_r, precio_r, total_r],
                        "Promedio del grupo": prom.values.round(2),
                    })
                    st.markdown("**Su compra frente al promedio del grupo**")
                    st.dataframe(comp, hide_index=True, width="stretch")

    # ------------------------------ Modo: pedido con varios productos
    else:
        with col_form:
            st.subheader("🛒 Arme el pedido")
            st.caption("Agregue cada producto con su cantidad. El precio se toma de la carta.")
            with st.container(border=True):
                producto = st.selectbox(
                    "Producto",
                    list(precios.keys()),
                    format_func=lambda it: f"{nombre_producto(it)} · ${precios[it]:,.2f}",
                )
                cant_prod = st.number_input("Cantidad", min_value=1, max_value=20, step=1, value=1)
                if st.button("➕ Agregar al pedido", width="stretch"):
                    for linea in st.session_state.pedido:
                        if linea["item"] == producto:
                            linea["cantidad"] += int(cant_prod)
                            break
                    else:
                        st.session_state.pedido.append({"item": producto, "cantidad": int(cant_prod)})
                    st.session_state.pedido_res = None

            if st.session_state.pedido:
                tabla = pd.DataFrame([
                    {
                        "Producto": nombre_producto(l["item"]),
                        "Cantidad": l["cantidad"],
                        "Precio unitario": precios[l["item"]],
                        "Subtotal": l["cantidad"] * precios[l["item"]],
                    }
                    for l in st.session_state.pedido
                ])
                st.markdown("**Su pedido**")
                st.dataframe(tabla, hide_index=True, width="stretch")
                st.metric("Total del pedido", f"${tabla['Subtotal'].sum():,.2f}")
                b1, b2 = st.columns([2, 1])
                clasificar_pedido = b1.button("🔍 Clasificar pedido", type="primary", width="stretch")
                if b2.button("🗑️ Vaciar", width="stretch"):
                    st.session_state.pedido = []
                    st.session_state.pedido_res = None
                    st.rerun()
                if clasificar_pedido:
                    cant = [l["cantidad"] for l in st.session_state.pedido]
                    prec = [precios[l["item"]] for l in st.session_state.pedido]
                    tot = [c * p for c, p in zip(cant, prec)]
                    grupos = model.predict(a_modelo(cant, prec, tot))
                    res = tabla.copy()
                    res["grupo"] = [int(g) for g in grupos]
                    res["Tipo de compra"] = res["grupo"].map(nombre_cluster)
                    st.session_state.pedido_res = res
                    hora = datetime.now().strftime("%H:%M:%S")
                    for _, fila in res.iterrows():
                        st.session_state.historial.insert(0, {
                            "Hora": hora,
                            "Producto": fila["Producto"],
                            "Cantidad": fila["Cantidad"],
                            "Precio unitario": fila["Precio unitario"],
                            "Gasto total": fila["Subtotal"],
                            "Tipo de compra": fila["Tipo de compra"],
                        })
            else:
                st.info("El pedido está vacío. Agregue al menos un producto.")

        with col_res:
            st.subheader("🎯 Resultado del pedido")
            res = st.session_state.pedido_res
            if res is None:
                st.info(
                    "Arme el pedido y presione **Clasificar pedido**.\n\n"
                    "El modelo aprendió con compras de **un producto a la vez**, así que cada "
                    "producto del pedido se clasifica por separado."
                )
            else:
                gasto_por_tipo = res.groupby("grupo")["Subtotal"].sum()
                principal = int(gasto_por_tipo.idxmax())
                tarjeta_resultado(principal, "Tipo de compra principal del pedido")
                st.caption("El tipo principal es el que concentra la mayor parte del dinero del pedido.")
                r1, r2, r3 = st.columns(3)
                r1.metric("Productos", len(res))
                r2.metric("Unidades", int(res["Cantidad"].sum()))
                r3.metric("Total", f"${res['Subtotal'].sum():,.2f}")
                st.markdown("**Cada producto del pedido**")
                st.dataframe(
                    res[["Producto", "Cantidad", "Subtotal", "Tipo de compra"]],
                    hide_index=True, width="stretch",
                )
                reparto = gasto_por_tipo.reset_index()
                reparto["Tipo de compra"] = reparto["grupo"].map(nombre_cluster)
                reparto["color"] = reparto["grupo"].map(lambda g: CLUSTER_INFO[int(g)]["color"])
                st.markdown("**¿Cómo se reparte el dinero del pedido?**")
                st.altair_chart(
                    alt.Chart(reparto)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Subtotal:Q", title="Dinero ($)"),
                        y=alt.Y("Tipo de compra:N", title=None, axis=alt.Axis(labelLimit=320)),
                        color=alt.Color("color:N", scale=None, legend=None),
                        tooltip=["Tipo de compra", alt.Tooltip("Subtotal:Q", title="Dinero ($)")],
                    )
                    .properties(height=40 + 40 * len(reparto)),
                    width="stretch",
                )

    if st.session_state.historial:
        st.divider()
        h1, h2 = st.columns([3, 1])
        h1.subheader("🕘 Historial de esta sesión")
        hist_df = pd.DataFrame(st.session_state.historial)
        h2.download_button(
            "⬇️ Descargar historial", hist_df.to_csv(index=False).encode("utf-8"),
            file_name="historial_clasificaciones.csv", mime="text/csv", width="stretch",
        )
        st.dataframe(hist_df, hide_index=True, width="stretch")
        if st.button("🗑️ Borrar historial"):
            st.session_state.historial = []
            st.rerun()

# ---------------------------------------------------------------- Clasificar archivo
with tab_lote:
    st.subheader("📁 Clasificar varias compras desde un archivo")
    st.markdown(
        "Suba un archivo **CSV** con las columnas **Cantidad** y **Precio unitario** "
        "(y opcionalmente **Gasto total**; si no está, se calcula). La aplicación asigna "
        "un tipo de compra a cada fila y permite descargar el resultado."
    )
    plantilla = pd.DataFrame(
        {"Cantidad": [1, 5, 4], "Precio unitario": [2.0, 1.0, 5.0], "Gasto total": [2.0, 5.0, 20.0]}
    )
    st.download_button(
        "⬇️ Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_compras.csv", mime="text/csv",
    )
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
            # Se aceptan columnas en español o con los nombres originales en inglés.
            lote = lote.rename(columns={k: v for k, v in VAR_ES.items() if k in lote.columns})
            if "Gasto total" not in lote.columns and {"Cantidad", "Precio unitario"} <= set(lote.columns):
                lote["Gasto total"] = lote["Cantidad"] * lote["Precio unitario"]
            faltan = [c for c in ["Cantidad", "Precio unitario"] if c not in lote.columns]
            if faltan:
                st.error(f"Faltan columnas en el archivo: {', '.join(faltan)}")
            else:
                grupos = model.predict(a_modelo(lote["Cantidad"], lote["Precio unitario"], lote["Gasto total"]))
                lote["Tipo de compra"] = [nombre_cluster(g) for g in grupos]
                st.success(f"Se clasificaron {len(lote)} compras.")
                resumen = lote["Tipo de compra"].value_counts().rename_axis("Tipo de compra").reset_index(name="Compras")
                c1, c2 = st.columns([1, 2])
                c1.dataframe(resumen, hide_index=True, width="stretch")
                c2.dataframe(lote, hide_index=True, width="stretch", height=260)
                st.download_button(
                    "⬇️ Descargar resultado", lote.to_csv(index=False).encode("utf-8"),
                    file_name="compras_clasificadas.csv", mime="text/csv",
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
                st.metric("Compras", f"{len(sub):,}".replace(",", "."))
                st.caption(f"{len(sub) / len(df):.0%} del total de compras")
                st.markdown(
                    f"- Cantidad promedio: **{sub['Quantity'].mean():.1f}** uds\n"
                    f"- Precio unitario promedio: **\\${sub['Price Per Unit'].mean():.2f}**\n"
                    f"- Gasto promedio: **\\${sub['Total Spent'].mean():.2f}**"
                )

    if df is not None:
        dominio = [nombre_cluster(c) for c in CLUSTER_INFO]
        colores = [CLUSTER_INFO[c]["color"] for c in CLUSTER_INFO]
        leyenda = alt.Legend(orient="bottom", title=None, labelLimit=320)

        st.markdown("#### ¿Cómo se ven los grupos?")
        st.caption("Cada punto es una combinación de cantidad y precio; el tamaño indica cuántas compras hay.")
        puntos = (
            df.groupby(["Quantity", "Price Per Unit", "cluster"]).size().reset_index(name="Compras")
            .rename(columns={"Quantity": "Cantidad", "Price Per Unit": "Precio unitario"})
        )
        puntos["Tipo de compra"] = puntos["cluster"].map(nombre_cluster)
        scatter = (
            alt.Chart(puntos)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X("Cantidad:Q", scale=alt.Scale(domain=[0, 6]), axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("Precio unitario:Q", title="Precio unitario ($)", scale=alt.Scale(domain=[0, 6]),
                        axis=alt.Axis(tickMinStep=1)),
                size=alt.Size("Compras:Q", legend=None),
                color=alt.Color("Tipo de compra:N", scale=alt.Scale(domain=dominio, range=colores), legend=leyenda),
                tooltip=["Tipo de compra", "Cantidad", "Precio unitario", "Compras"],
            )
            .properties(height=380)
        )
        st.altair_chart(scatter, width="stretch")

        st.markdown("#### Productos más vendidos en cada tipo de compra")
        prod = df.groupby(["cluster", "Item"]).size().reset_index(name="Compras")
        prod["Producto"] = prod["Item"].map(nombre_producto)
        prod["Tipo de compra"] = prod["cluster"].map(nombre_cluster)
        barras = (
            alt.Chart(prod)
            .mark_bar()
            .encode(
                x=alt.X("Compras:Q"),
                y=alt.Y("Producto:N", title=None, sort="-x", axis=alt.Axis(labelOverlap=False)),
                color=alt.Color("Tipo de compra:N", scale=alt.Scale(domain=dominio, range=colores), legend=leyenda),
                tooltip=["Tipo de compra", "Producto", "Compras"],
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
            "Cada fila es la compra de un producto."
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Filas", f"{len(df):,}".replace(",", "."))
        d2.metric("Productos distintos", df["Item"].nunique())
        d3.metric("Gasto promedio", f"${df['Total Spent'].mean():.2f}")
        vista = df.drop(columns=["cluster"]).head(200).copy()
        vista["Item"] = vista["Item"].map(nombre_producto)
        vista["Payment Method"] = vista["Payment Method"].map(PAGO_ES).fillna(vista["Payment Method"])
        vista["Location"] = vista["Location"].map(LUGAR_ES).fillna(vista["Location"])
        st.dataframe(vista.rename(columns=COLUMNAS_ES), hide_index=True, width="stretch", height=320)
        st.caption("Se muestran las primeras 200 filas.")
        with st.expander("⚠️ Limitaciones de los datos"):
            st.markdown(
                "- El gasto total debería ser cantidad × precio, pero no coincide en ~11% de las filas.\n"
                "- Algunas filas tienen precio 3.0 en productos que no cuestan eso; probablemente "
                "son valores faltantes rellenados con la mediana.\n"
                "- Cantidad y precio toman pocos valores (1 a 5), por lo que los grupos dividen una "
                "\"cuadrícula\" más que descubrir grupos naturales.\n"
                "- Cada fila es la compra de un producto, no un cliente: los grupos son tipos de compra."
            )
    else:
        st.info("El archivo de datos no está disponible en este despliegue.")

# ---------------------------------------------------------------- Cómo funciona
with tab_como:
    st.subheader("ℹ️ ¿Cómo funciona?")
    st.markdown(
        """
        1. **Entrenamiento (notebook y `train.py`):** con la cantidad, el precio unitario y
           el gasto total de cada compra se entrena un modelo de scikit-learn:
           imputación por mediana → estandarización → **K-Means con 3 grupos**
           (con semilla fija para que el resultado sea reproducible).
        2. **Modelo guardado:** el modelo entrenado se guarda en `model.joblib` junto con
           las variables y la descripción de cada grupo.
        3. **Esta aplicación:** carga `model.joblib` **sin volver a entrenar**, aplica las
           mismas transformaciones del entrenamiento y asigna el grupo más cercano.
        4. **Pedidos con varios productos:** como el modelo aprendió con compras de un
           producto a la vez, cada producto del pedido se clasifica por separado y se
           muestra el tipo de compra que concentra más dinero.
        5. **Despliegue:** la aplicación se empaqueta con **Docker** (`Dockerfile` →
           `docker build` → imagen → `docker run` → contenedor).
        """
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
    '<div class="footer">Proyecto Reto ML · Streamlit + Docker · Ruta B: agrupamiento con K-Means<br>'
    "Datos: Kaggle · Ventas de café de dos tiendas</div>",
    unsafe_allow_html=True,
)
