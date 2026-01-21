import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# Conexão VPS (Porta 5433 conforme sua imagem do Easypanel)
PG_CONN = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)
register_vector(PG_CONN)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def ingest_tax_facts():
    # Baseado na imagem da matriz interestadual e regras de 2026
    # Alíquotas interestaduais: 7% (Sul/Sudeste -> Norte/Nordeste/CO) ou 12% (Demais)
    tax_facts = [
        # REGRAS GERAIS INTERESTADUAIS
        {"fact": "Operações interestaduais saindo de SP, RJ, MG, PR, RS, SC com destino a estados das regiões Norte, Nordeste, Centro-Oeste e ES possuem alíquota de 7%.", "tags": "interestadual, 7%"},
        {"fact": "Operações interestaduais entre estados da mesma região ou saindo do N/NE/CO para o S/SE possuem alíquota de 12%.", "tags": "interestadual, 12%"},
        {"fact": "Produtos importados ou com conteúdo de importação superior a 40% possuem alíquota interestadual unificada de 4% (Resolução 13/2012).", "tags": "importados, 4%"},
        
        # ALÍQUOTAS INTERNAS E SUPÉRFLUOS (EXEMPLOS)
        {"fact": "Em Santa Catarina (SC), a alíquota interna padrão é 17%. Itens supérfluos como bebidas alcoólicas e tabaco possuem alíquota de 25%.", "uf": "SC"},
        {"fact": "No Rio de Janeiro (RJ), a alíquota padrão é 20% (incluindo FECOP). Itens supérfluos podem chegar a 32%.", "uf": "RJ"},
        {"fact": "Em São Paulo (SP), a alíquota interna é 18%. Operações com perfumes e cosméticos (supérfluos) possuem alíquota de 25%.", "uf": "SP"}
    ]

    with PG_CONN.cursor() as cur:
        for item in tax_facts:
            content = item['fact']
            embedding = model.encode(content).tolist()
            metadata = {"tipo": "matriz_icms", "origem": "tabela_oficial", "contexto": "sistema_antigo"}
            
            cur.execute(
                "INSERT INTO legal_vectors (id, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), content, embedding, str(metadata))
            )
    PG_CONN.commit()
    print("✅ Matriz ICMS e Fatos de Supérfluos injetados na VPS!")

if __name__ == "__main__":
    ingest_tax_facts()