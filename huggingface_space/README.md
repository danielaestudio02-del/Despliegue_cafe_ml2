---
title: Cafe ML2 Segmentacion
emoji: ☕
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Café ML2 — Segmentación de transacciones de café

Aplicación Streamlit que asigna una transacción de venta de café a uno de los 3 tipos de compra encontrados con K-Means (aprendizaje no supervisado).

Se construye con el mismo `Dockerfile` del repositorio del proyecto: https://github.com/danielaestudio02-del/Despliegue_cafe_ml2

## Ejemplos para probar

| Cantidad | Precio unitario | Gasto total | Cluster esperado |
|---|---|---|---|
| 1 | 2 | 2 | 2 |
| 5 | 1 | 5 | 1 |
| 4 | 5 | 20 | 0 |
