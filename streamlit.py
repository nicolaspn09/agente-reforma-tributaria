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

# --- SETUP E CACHE DE MODELO ---
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv(os.path.join(script_dir, '.env')))
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@st.cache_resource
def load_ai_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

model = load_ai_models()

# --- 1. CONEXÕES COM A VPS ---
ES_CLIENT = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
)

PG_CONN = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)
register_vector(PG_CONN)

# --- 2. FERRAMENTAS ESTRATÉGICAS (Otimizadas para Economia) ---
class ConsultorMasterTools:
    @tool("consultar_inteligencia_vps")
    def consultar_inteligencia_vps(query: str):
        """Consulta prioritária na base local (LCP 214, EC 132 e ICMS). Use sempre esta ferramenta primeiro."""
        termos_map = {"aluguel": "locação de bens imóveis", "nota de débito": "reembolso", "software": "licenciamento"}
        for k, v in termos_map.items():
            if k in query.lower(): query += f" {v}"

        query_vector = model.encode(query).tolist()
        with PG_CONN.cursor() as cur:
            # Selecionamos 'content' e 'metadata' para transparência
            cur.execute("SELECT content, metadata FROM legal_vectors ORDER BY embedding <=> %s::vector LIMIT 8", (query_vector,))
            res = cur.fetchall()
        
        contexto = "### DADOS DA VPS (FONTE INTERNA) ###\n"
        for r in res:
            content, metadata = r
            contexto += f"\n- [REF: {metadata}] {content}"
        return contexto

    @tool("pesquisar_planalto_web")
    def pesquisar_planalto_web(query: str):
        """Busca na Web (Planalto/Sefaz). USE APENAS se a base VPS não tiver a resposta ou para notícias de última hora."""
        # Reduzido para 3 resultados (k=3) para economizar tokens e créditos
        search = TavilySearchResults(
            k=3, 
            include_domains=["planalto.gov.br", "reformatributaria.com.br", "sefaz.gov.br", "fazenda.gov.br"]
        )
        return search.run(query)

# --- 3. FUNÇÃO DE PARECER EM PDF ---
def gerar_pdf(texto, query_origem):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "Parecer Tecnico Estrategico - Auditoria AI", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.ln(5)
    pdf.multi_cell(0, 7, f"Analise solicitada: {query_origem}")
    pdf.ln(5)
    # Limpa markdown para o PDF
    clean_text = texto.replace("#", "").replace("*", "").replace("|", "-")
    pdf.multi_cell(0, 7, clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())

# --- 4. INTERFACE E AGENTE ---
st.set_page_config(page_title="Consultor Reforma AI", layout="wide", page_icon="⚖️")
st.title("⚖️ Agente Consultor Estratégico - Reforma Tributária")

user_query = st.text_area("Descreva sua dúvida ou operação de negócio:", 
                          placeholder="Ex: Qual a alíquota de SP para MG?")

if st.button("Gerar Parecer Autônomo"):
    if user_query:
        with st.spinner("Consultando base de conhecimento (VPS)..."):
            
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.environ["GROQ_API_KEY"])

            agente_master = Agent(
                role='Consultor Tributário Master',
                goal='Analisar a Reforma Tributária com foco em economia de recursos e transparência de fontes.',
                backstory="""Você é um auditor rigoroso. 
                Sua prioridade número 1 é usar a base VPS (interna). 
                Você só deve usar a pesquisa Web se a informação for inexistente na VPS.
                Ao responder, você DEVE citar explicitamente a fonte (Artigo da lei ou link do site).""",
                tools=[ConsultorMasterTools.consultar_inteligencia_vps, ConsultorMasterTools.pesquisar_planalto_web],
                llm=llm,
                verbose=True,
                max_iter=4, # Reduzido de 5 para 4 para economizar tokens/tempo
                memory=True
            )

            task = Task(
                description=f"""Analise a solicitação: '{user_query}'.
                1. Procure primeiro na VPS as alíquotas interestaduais ou artigos da LCP 214.
                2. Somente se necessário, busque no Planalto Web.
                3. Estruture a resposta citando as fontes de cada dado.
                4. Explique o impacto estratégico (Crédito do Art. 121).""",
                expected_output="Parecer Técnico claro, com fontes citadas e comparativo de alíquotas.",
                agent=agente_master
            )

            crew = Crew(agents=[agente_master], tasks=[task])
            resultado = str(crew.kickoff())
            
            st.session_state['res'] = resultado
            st.session_state['query'] = user_query
            st.markdown(resultado)

if 'res' in st.session_state:
    pdf_bytes = gerar_pdf(st.session_state['res'], st.session_state['query'])
    st.download_button("📥 Baixar Parecer em PDF", data=pdf_bytes, file_name="parecer_tributario.pdf", mime="application/pdf")
