import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from dotenv import load_dotenv, find_dotenv


script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv(os.path.join(script_dir, '.env')))

# 1. Configuração da Tool de Busca focada na ROIT
# Isso garante que o agente não busque em blogs genéricos, mas na fonte técnica
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class RoitSearchTool:
    @tool("search_roit_portal")
    def search_roit_portal(query: str):
        """Busca análises técnicas e notícias recentes exclusivamente no portal reformatributaria.com.br."""
        search = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_domains=["reformatributaria.com.br"]
        )
        return search.run(query)

# 2. Definição do Agente Auditor
auditor_fiscal = Agent(
    role='Auditor Fiscal Sênior - Reforma Tributária',
    goal='Analisar impactos tributários de produtos e serviços com base na LCP 214/2025 e EC 132.',
    backstory="""Especialista em compliance com 12 anos de experiência. 
    Sua missão é cruzar a letra fria da lei (PDFs) com as interpretações técnicas da ROIT.
    Você é rigoroso, nunca inventa alíquotas e sempre cita os Artigos.""",
    tools=[RoitSearchTool.search_roit_portal], # Aqui adicionaremos a Tool de RAG do Postgres/ES depois
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm="groq/llama-3.3-70b-versatile" # Recomendado para raciocínio jurídico complexo
)

# 3. Definição da Tarefa de Análise
def criar_tarefa_analise(produto_ou_duvida):
    return Task(
        description=f"""Analise o seguinte: {produto_ou_duvida}.
        1. Identifique o enquadramento na LCP 214/2025 (Cesta Básica, Redução 60% ou Alíquota Padrão).
        2. Verifique no portal da ROIT se há decisões recentes sobre este item.
        3. Retorne no formato: Veredito, Base Legal e Contexto ROIT.""",
        expected_output="Um relatório técnico detalhado com citações de artigos e fundamentação da ROIT.",
        agent=auditor_fiscal
    )

# Exemplo de Orquestração
crew = Crew(agents=[auditor_fiscal], tasks=[criar_tarefa_analise("Alíquota para Carnes e Proteínas")], process=Process.sequential)
resultado = crew.kickoff()