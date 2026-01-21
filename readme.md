# 🏛️ Auditor Fiscal Autônomo - Reforma Tributária 2026

Este projeto implementa um **Agente de IA Consultivo** especializado na Reforma Tributária Brasileira (LCP 214/2025 e EC 132). O sistema utiliza uma arquitetura de **RAG Híbrido** (Retrieval-Augmented Generation) para fornecer análises técnicas precisas e comparativos de carga tributária.

## 🚀 Diferenciais Tecnológicos
- **RAG Híbrido:** Integração entre busca semântica (PostgreSQL + pgvector) e busca por palavras-chave (Elasticsearch).
- **Raciocínio Agêntico:** Orquestração via `CrewAI` para interpretar intenções de usuários sem a necessidade de inputs técnicos rígidos (NCM/CEST).
- **Inteligência Federativa:** Matriz de alíquotas interestaduais injetada para simulação real de impacto de ICMS vs. novo IVA Dual.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Agent Framework:** CrewAI
- **LLM:** Groq (Llama 3.3 70B)
- **Bancos de Dados:** PostgreSQL (pgvector) e Elasticsearch
- **Embeddings:** SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)

## 🐳 Como Executar (Docker Compose)
1. Clone o repositório.
2. Configure o arquivo `.env` com suas credenciais.
3. Execute o comando:
   ```bash
   docker-compose up --build -d

---

### 🚀 Subindo na VPS Hostinger via Docker Compose

Usar o Docker Compose é a maneira mais rápida e profissional de liberar o acesso para sua amiga.

#### 1. Preparar os Arquivos
No seu computador, crie um arquivo chamado `requirements.txt` com todas as bibliotecas usadas (streamlit, crewai, elasticsearch, psycopg2-binary, pgvector, sentence-transformers, torch, fpdf2, python-dotenv).



#### 2. Criar o Dockerfile
Crie um arquivo `Dockerfile` na raiz do projeto:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]