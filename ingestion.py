import os
import re
import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import psycopg2
from pgvector.psycopg2 import register_vector
import urllib3
from dotenv import load_dotenv, find_dotenv

# Silencia o aviso de InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# --- CONFIGURAÇÃO DE COMPATIBILIDADE FORÇADA ---
ES_CLIENT = Elasticsearch(
    ES_HOST, # Porta 443 é padrão para HTTPS no Easypanel
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    request_timeout=60,
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/json"
    }
)

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

def setup_database():
    """Garante que a extensão de vetor e a tabela existam na VPS."""
    with PG_CONN.cursor() as cur:
        # 1. Ativa a extensão pgvector no banco 'rpa'
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Cria a tabela com a dimensão correta (384 para o modelo MiniLM-L12)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS legal_vectors (
                id UUID PRIMARY KEY,
                content TEXT,
                embedding VECTOR(384),
                metadata TEXT
            );
        """)
    PG_CONN.commit()
    print("✅ Banco de dados preparado: Extensão e Tabela 'legal_vectors' prontas.")

# Modelo Multilíngue (Excelente para Português Jurídico)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_hierarchical_context(text):
    """Extrai o Artigo atual para servir de metadado pai."""
    art_match = re.search(r"(Art\.\s?\d+[ºA-Z]?)", text)
    return art_match.group(1) if art_match else "Contexto Geral"

def process_and_ingest(pdf_path, index_name="reforma_idx"):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # Splitter que respeita a quebra de Artigos e Parágrafos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nArt.", "\n§", "\nI -", "\nII -", "\n\n", " "]
    )
    
    chunks = splitter.split_documents(docs)
    
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        contexto_pai = get_hierarchical_context(chunk.page_content)
        embedding = model.encode(chunk.page_content).tolist()
        
        metadata = {
            "source": pdf_path,
            "artigo_pai": contexto_pai,
            "page": chunk.metadata.get("page")
        }

        # 1. Ingestão no PGVector (Busca Semântica)
        with PG_CONN.cursor() as cur:
            cur.execute(
                "INSERT INTO legal_vectors (id, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                (chunk_id, chunk.page_content, embedding, str(metadata))
            )
        
        # 2. Ingestão no Elasticsearch (Busca por Keyword/Termo Técnico)
        doc_es = {
            "content": chunk.page_content,
            "metadata": metadata,
            "text_vector": embedding # ES também permite busca vetorial se desejar
        }
        ES_CLIENT.index(index=index_name, id=chunk_id, document=doc_es)
    
    PG_CONN.commit()
    print(f"Sucesso: {len(chunks)} fragmentos processados do arquivo {pdf_path}")


# Chame a função antes de processar os PDFs
setup_database()

# Execução inicial
process_and_ingest(r"C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Agente Reforma Tributária - Projeto\Legislação\Lcp 214.pdf")
process_and_ingest(r"C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Agente Reforma Tributária - Projeto\Legislação\Emenda Constitucional nº 132.pdf")