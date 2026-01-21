FROM python:3.11-slim

# 1. Instala dependências de compilação essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instala primeiro as bibliotecas pesadas separadamente para usar cache
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers

# 3. Instala o restante das dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 4. Expõe a porta do Streamlit
EXPOSE 8501

# 5. Comando de inicialização otimizado para produção
CMD ["streamlit", "run", "streamlit.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]