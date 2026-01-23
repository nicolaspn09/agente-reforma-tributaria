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
    # Evita erro de Meta Tensor forçando o dispositivo correto
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

model = load_ai_models()

# --- 1. CONEXÕES COM A VPS (BANCO DE DADOS SOBERANO) ---
ES_CLIENT = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
)

PG_CONN = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port="5433", # Porta específica do pgvector
    database=os.getenv("PG_DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)
register_vector(PG_CONN)

# --- 2. FERRAMENTAS ESTRATÉGICAS (TOOLS) ---
class ConsultorMasterTools:
    @tool("consultar_inteligencia_vps")
    def consultar_inteligencia_vps(query: str):
        """Consulta LCP 214/2025, EC 132 e Matriz ICMS interestadual. 
        Mapeia automaticamente sinônimos como 'aluguel' para 'locação'."""
        
        # Guardrail de Sinônimos: Transforma termos comerciais em termos jurídicos
        termos_map = {"aluguel": "locação de bens imóveis", "nota de débito": "reembolso", "software": "licenciamento de bens imateriais"}
        for k, v in termos_map.items():
            if k in query.lower(): query += f" {v}"

        query_vector = model.encode(query).tolist()
        with PG_CONN.cursor() as cur:
            # Busca ampliada para garantir que pegue a lei e a matriz interestadual
            cur.execute("SELECT content FROM legal_vectors ORDER BY embedding <=> %s::vector LIMIT 10", (query_vector,))
            res = cur.fetchall()
        
        return "### CONHECIMENTO INTERNO (VPS) ###\n" + "\n".join([f"- {r[0]}" for r in res])

    @tool("pesquisar_planalto_e_governo")
    def pesquisar_planalto_e_governo(query: str):
        """Busca no site do Planalto, Portais da Fazenda e notícias oficiais da Reforma."""
        search = TavilySearchResults(
            k=5, 
            include_domains=["planalto.gov.br", "reformatributaria.com.br", "sefaz.gov.br", "fazenda.gov.br"]
        )
        return search.run(query)

# --- 3. FUNÇÃO DE PARECER EM PDF ---
def gerar_pdf(texto, query_origem):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "Parecer Tecnico - Consultoria Estrategica Reforma", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.ln(5)
    pdf.multi_cell(0, 7, f"Analise solicitada: {query_origem}")
    pdf.ln(5)
    # Limpa markdown para o PDF não quebrar
    clean_text = texto.replace("#", "").replace("*", "").replace("|", "-")
    pdf.multi_cell(0, 7, clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())

# --- 4. INTERFACE E LÓGICA DO AGENTE ---
st.set_page_config(page_title="Consultor Reforma AI", layout="wide", page_icon="⚖️")
st.title("⚖️ Agente Consultor Estratégico - Reforma Tributária")
st.markdown("Analise qualquer operação (Bens, Serviços, Locações ou Direitos) comparando o sistema atual com a LCP 214/2025.")

user_query = st.text_area("Descreva sua dúvida ou operação de negócio:", 
                          placeholder="Ex: Impacto do aluguel comercial em SP | Venda de SC para RJ | Como funciona o cashback?")

if st.button("Gerar Parecer Autônomo"):
    if user_query:
        with st.spinner("O Agente está cruzando a Matriz Interestadual com a Legislação do Planalto..."):
            
            # Inicialização do Cérebro (Llama 3.3 via Groq)
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.environ["GROQ_API_KEY"])

            agente_master = Agent(
                role='Consultor Tributário Sênior Master',
                goal='Fornecer análises técnicas irrepreensíveis e estratégicas sobre a Reforma Tributária brasileira.',
                backstory="""Você é a maior autoridade em planejamento tributário. 
                Sua especialidade é o IVA Dual (IBS/CBS). Você nunca inventa dados. 
                Se o usuário fala 'aluguel', você sabe que a lei trata como 'locação'. 
                Você usa a matriz de ICMS na VPS para ser exato no sistema antigo.""",
                tools=[ConsultorMasterTools.consultar_inteligencia_vps, ConsultorMasterTools.pesquisar_planalto_e_governo],
                llm=llm,
                verbose=True,
                max_iter=5, # Limite de reflexão para evitar loop
                memory=True
            )

            task = Task(
                description=f"""Analise tecnicamente: '{user_query}'.
                1. Identifique o objeto: é um BEM, SERVIÇO, DIREITO ou LOCAÇÃO?
                2. Extraia Origem/Destino e use a Ferramenta VPS para pegar a alíquota interestadual EXATA da matriz de ICMS.
                3. Na LCP 214/2025, identifique o regime: Alíquota Zero, Redução ou Alíquota Padrão (26,5%).
                4. Explique o conceito de Crédito Pleno (Art. 121) e o fim da cumulatividade no cenário descrito.""",
                expected_output="Parecer Técnico com Diagnóstico, Base Legal e Visão Estratégica.",
                agent=agente_master
            )

            crew = Crew(agents=[agente_master], tasks=[task])
            resultado = str(crew.kickoff())
            
            st.session_state['res'] = resultado
            st.session_state['query'] = user_query
            st.markdown(resultado)

if 'res' in st.session_state:
    pdf_bytes = gerar_pdf(st.session_state['res'], st.session_state['query'])
    st.download_button("📥 Baixar Parecer em PDF", data=pdf_bytes, file_name="parecer_tecnico_reforma.pdf", mime="application/pdf")
