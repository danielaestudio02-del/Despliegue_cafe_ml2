FROM python:3.12-slim

WORKDIR /app

# Carpeta con permisos de escritura para la configuración de Streamlit,
# aunque el contenedor no se ejecute como administrador (p. ej. Hugging Face).
ENV HOME=/tmp

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
