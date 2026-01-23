import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import uuid
import os
from dotenv import load_dotenv, find_dotenv

# --- CONFIGURAÇÕES ---
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv(os.path.join(script_dir, '.env')))

# Conexão com a VPS (Porta 5433 do pgvector)
PG_CONN = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)
register_vector(PG_CONN)

# Modelo Multilíngue
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def ingest_matrix(file_path):
    # 1. Carrega o CSV tratando o separador ';' e definindo a primeira coluna como índice
    df = pd.read_csv(file_path, sep=';', index_col=0)
    
    # 2. LIMPEZA: Remove a coluna 'destino' e a linha 'origem' se existirem
    if 'destino' in df.columns:
        df = df.drop(columns=['destino'])
    if 'origem' in df.index:
        df = df.drop(index='origem')

    print(f"🚀 Iniciando a injeção de {df.size} combinações de alíquotas na VPS...")
    
    contador = 0
    with PG_CONN.cursor() as cur:
        for origem, row in df.iterrows():
            for destino, aliquota in row.items():
                # Ignora valores vazios ou colunas de metadados
                if pd.isna(aliquota):
                    continue
                
                origem_uf = str(origem).upper()
                destino_uf = str(destino).upper()
                aliquota_fmt = f"{aliquota}%"

                # 3. CRIAÇÃO DO FATO (Lógica Interna vs Interestadual)
                if origem_uf == destino_uf:
                    texto = f"A alíquota interna padrão de ICMS no estado de {origem_uf} é de {aliquota_fmt}."
                else:
                    texto = f"A alíquota interestadual de ICMS em operações saindo de {origem_uf} com destino a {destino_uf} é de {aliquota_fmt}."
                
                # Gerar o vetor e metadados
                embedding = model.encode(texto).tolist()
                metadata = {
                    "tipo": "matriz_icms",
                    "origem": origem_uf,
                    "destino": destino_uf,
                    "aliquota": aliquota_fmt
                }
                
                # Inserção no Banco
                cur.execute(
                    "INSERT INTO legal_vectors (id, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                    (str(uuid.uuid4()), texto, embedding, str(metadata))
                )
                contador += 1
                
                # Print de progresso a cada 100 registros
                if contador % 100 == 0:
                    print(f"📡 {contador} registros processados...")

    PG_CONN.commit()
    print(f"✅ Sucesso: {contador} alíquotas interestaduais integradas à inteligência do Agente!")

if __name__ == "__main__":
    ingest_matrix(r"C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Agente Reforma Tributária - Projeto\Alíquota interestadual - data-1769102722017.csv")
