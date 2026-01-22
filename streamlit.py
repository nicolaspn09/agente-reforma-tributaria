import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain.tools import tool
from elasticsearch import Elasticsearch
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_community.tools.tavily_search import TavilySearchResults
import urllib3
import torch
from langchain_groq import ChatGroq
from fpdf import FPDF
from dotenv import load_dotenv, find_dotenv

# --- SETUP E CACHE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv(os.path.join(script_dir, '.env')))
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@st.cache_resource
def load_ai_models():
    # Forçar CPU se houver erro de Meta Tensor em ambientes específicos
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

model = load_ai_models()

# --- 1. CONEXÕES VPS ---
ES_CLIENT = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
)

PG_CONN = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port="5433", # Porta do pgvector
    database=os.getenv("PG_DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)
register_vector(PG_CONN)

# --- 2. FERRAMENTAS ESTRATÉGICAS ---
class ConsultorTools:
    @tool("consultar_base_conhecimento")
    def consultar_base_conhecimento(query: str):
        """Consulta a LCP 214/2025, EC 132 e a Matriz de ICMS (interestadual/interna) na VPS."""
        query_vector = model.encode(query).tolist()
        with PG_CONN.cursor() as cur:
            # Busca ampliada para capturar fatos da matriz e da lei
            cur.execute("SELECT content, metadata FROM legal_vectors ORDER BY embedding <=> %s::vector LIMIT 6", (query_vector,))
            pg_res = cur.fetchall()
        
        contexto = "### FATOS LEGAIS E TRIBUTÁRIOS (BASE VPS) ###\n"
        for r in pg_res:
            contexto += f"\n- {r[0]}"
        return contexto

    @tool("pesquisar_atualidades_web")
    def pesquisar_atualidades_web(query: str):
        """Busca notícias recentes e interpretações técnicas em portais especializados da reforma."""
        search = TavilySearchResults(k=5, include_domains=["reformatributaria.com.br", "sefaz.gov.br", "portaltributario.com.br"])
        return search.run(query)

# --- 3. FUNÇÃO AUXILIAR PDF ---
def gerar_pdf(texto, query_usuario):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "Parecer Tecnico - Auditoria Independente", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.ln(5)
    pdf.multi_cell(0, 7, f"Consulta: {query_usuario}")
    pdf.ln(5)
    # Limpeza básica de markdown para PDF
    clean_text = texto.replace("#", "").replace("*", "").replace("|", "-")
    pdf.multi_cell(0, 7, clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())

# --- 4. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agente Tributário", layout="wide", page_icon="⚖️")
st.title("⚖️ Agente Consultor Estratégico - Reforma Tributária")

user_query = st.text_area("Descreva o cenário tributário ou produto:", 
                          placeholder="Ex: Qual o impacto de vender desodorante de SP para SC? ou Explique o Imposto Seletivo para bebidas.")

if st.button("Executar Análise Autônoma"):
    if user_query:
        with st.spinner("Analisando intenção, estados e legislação..."):
            # Inicialização explícita para evitar erro de validação (Pydantic fix)
            llm_consultor = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=os.environ["GROQ_API_KEY"]
            )

            agente_independente = Agent(
                role='Consultor Tributário Master',
                goal='Analisar impactos da LCP 214/2025, diferenciando itens essenciais de supérfluos.',
                backstory="""Especialista em Seletividade Fiscal. Você identifica automaticamente 
                se o item sofrerá Imposto Seletivo ou se terá Alíquota Zero (Art. 120). 
                Você cruza a matriz de ICMS da VPS com o novo IVA Dual.""",
                tools=[ConsultorTools.consultar_base_conhecimento, ConsultorTools.pesquisar_atualidades_web],
                llm=llm_consultor, # Passando o objeto instanciado
                verbose=True,
                allow_delegation=False
            )

            task = Task(
                description=f"""Analise a solicitação: '{user_query}'.
                1. Identifique o produto e os estados de ORIGEM e DESTINO.
                2. Verifique se o item é ESSENCIAL (Art. 120 - Alíquota Zero) ou SUPÉRFLUO (Art. 393 - Imposto Seletivo).
                3. Compare o ICMS interestadual atual (baseado na matriz da VPS) com o novo IVA Dual de 26,5%.
                4. Explique o benefício do crédito pleno no destino conforme o Art. 121.""",
                expected_output="Parecer técnico com Tabela Comparativa (Origem vs Destino) e Veredito de Seletividade.",
                agent=agente_independente
            )

            crew = Crew(agents=[agente_independente], tasks=[task])
            resultado = str(crew.kickoff())
            
            st.session_state['resultado'] = resultado
            st.session_state['query'] = user_query
            st.markdown(resultado)

if 'resultado' in st.session_state:
    pdf_bytes = gerar_pdf(st.session_state['resultado'], st.session_state['query'])
    st.download_button("📥 Baixar Parecer Técnico (PDF)", data=pdf_bytes, file_name="parecer_tributario.pdf", mime="application/pdf")
