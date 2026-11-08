import joblib
import pandas as pd
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Café ML2 - Segmentación de Compras",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Segmentación de Ventas de Café")
st.markdown(
    """
Esta aplicación clasifica en tiempo real una transacción de venta de café 
en uno de los **3 tipos de compra (clusters)** identificados mediante Machine Learning no supervisado (*K-Means*).
"""
)

# Cargar artefacto del modelo
try:
    artifact = joblib.load("model.joblib")
    model = artifact["model"]
    features = artifact["features"]
    numeric_features = artifact.get("numeric_features", [])
    task = artifact.get("task", "clustering")
    cluster_descriptions = artifact.get("cluster_descriptions", {})
except Exception as e:
    st.error(f"Error al cargar model.joblib: {e}")
    st.stop()

st.subheader("📋 Ingrese los datos de la transacción")

# Formulario de entrada
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input(
            "Cantidad de unidades (Quantity):",
            min_value=1.0,
            max_value=20.0,
            value=2.0,
            step=1.0,
            help="Número de artículos adquiridos en la compra (1 a 5 típicamente)",
        )
        price_per_unit = st.number_input(
            "Precio unitario (Price Per Unit) [$]:",
            min_value=0.5,
            max_value=20.0,
            value=3.0,
            step=0.5,
            help="Precio unitario del producto adquirido",
        )
    with col2:
        total_default = float(quantity * price_per_unit)
        total_spent = st.number_input(
            "Gasto total (Total Spent) [$]:",
            min_value=0.5,
            max_value=100.0,
            value=total_default,
            step=0.5,
            help="Gasto total facturado en la transacción",
        )

    submitted = st.form_submit_button("🔍 Clasificar Tipo de Compra", use_container_width=True)

if submitted:
    input_data = {
        "Quantity": quantity,
        "Price Per Unit": price_per_unit,
        "Total Spent": total_spent,
    }
    X_new = pd.DataFrame([input_data], columns=features)
    
    try:
        cluster_pred = int(model.predict(X_new)[0])
        
        st.success(f"### 🎯 Cluster Asignado: **Cluster {cluster_pred}**")
        
        # Descripción detallada según el artefacto
        desc = cluster_descriptions.get(cluster_pred, "Sin descripción disponible.")
        st.info(f"**Perfil del grupo:** {desc}")
        
        # Resumen intuitivo
        interpretaciones = {
            0: "🛍️ **Compra grande de productos de mayor valor** (ej. ensaladas, smoothies, sándwiches en volumen).",
            1: "🍪 **Compra de volumen con productos económicos** (ej. varias unidades de café, galletas, té).",
            2: "☕ **Compra individual / de paso** (ej. consumo ligero de una o dos unidades)."
        }
        if cluster_pred in interpretaciones:
            st.markdown(f"**Interpretación comercial:** {interpretaciones[cluster_pred]}")
            
    except Exception as e:
        st.error(f"Error al predecir: {e}")

st.markdown("---")
st.caption("Proyecto Reto ML: Streamlit + Docker + Compute Engine (GCP) | Ruta B: K-Means Clustering")
