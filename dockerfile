# Usar Python 3.12 slim para compatibilidad con xgboost 3.0.4
FROM python:3.12-slim

# Evitar prompts al instalar paquetes del sistema
ENV DEBIAN_FRONTEND=noninteractive

# Carpeta de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para algunos paquetes (XGBoost, numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Exponer puerto (opcional, para FastAPI)
EXPOSE 8000

# Comando por defecto para servir la API con uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
