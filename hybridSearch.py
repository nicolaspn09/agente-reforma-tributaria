import os
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain.tools import tool
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv, find_dotenv


script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv(os.path.join(script_dir, '.env')))

ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")
PG_DB_NAME = os.getenv("PG_DB_NAME")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")


# Configurações de Conexão (Ajuste para sua VPS)
ES_CLIENT = Elasticsearch(f"{ES_HOST}:9200")

# Parâmetros extraídos do seu print do Easypanel
PG_CONFIG = {
    "dbname": PG_DB_NAME,
    "user": PG_USER,
    "password": PG_PASSWORD,
    "host": PG_HOST,
    "port": PG_PORT
}

# String de conexão SQLAlchemy ou Psycopg2
CONNECTION_STRING = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"

PG_CONN = psycopg2.connect(CONNECTION_STRING)
register_vector(PG_CONN)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class HybridTaxRetriever:
    
    @tool("consultar_legislacao_reforma")
    def search(query: str):
        """
        Consulta a LCP 214/2025 e a EC 132 usando busca híbrida.
        Ideal para encontrar alíquotas, NCMs e regras de crédito.
        """
        query_vector = model.encode(query).tolist()
        
        # 1. Busca Semântica (PGVector) - Top 5
        with PG_CONN.cursor() as cur:
            cur.execute("""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as score 
                FROM legal_vectors 
                ORDER BY embedding <=> %s::vector LIMIT 5
            """, (query_vector, query_vector))
            pg_results = cur.fetchall()

        # 2. Busca por Keyword (Elasticsearch BM25) - Top 5
        es_query = {
            "query": {
                "match": {"content": query}
            },
            "size": 5
        }
        es_res = ES_CLIENT.search(index="reforma_idx", body=es_query)
        es_results = es_res['hits']['hits']

        # 3. Lógica de Fusão Simples (Priorizando Intersecção)
        # Em um cenário real, usaríamos RRF, aqui vamos concatenar com prioridade
        context = "--- RESULTADOS DA LEGISLAÇÃO ---\n"
        
        for res in pg_results:
            context += f"[Fonte: {res[1]['artigo_pai']}] {res[0]}\n\n"
            
        for hit in es_results:
            # Evita duplicidade simples
            if hit['_source']['content'][:100] not in context:
                context += f"[Keyword Match: {hit['_source']['metadata']['artigo_pai']}] {hit['_source']['content']}\n\n"

        return context

# Adicione esta Tool ao seu Agente criado no passo anterior:
# auditor_fiscal.tools.append(HybridTaxRetriever.search)