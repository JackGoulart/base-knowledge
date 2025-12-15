"""
Agente de Busca DuckDuckGo - Agente WEBSEARCH que trata consultas que requerem informações atualizadas da web.
Implementação avançada usando agentes LangChain com ferramentas baseadas em MCP para máxima flexibilidade.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.agents import create_agent
from services.mcp_search_client import get_mcp_search_client
from services.openai_client import create_chat_llm

logger = logging.getLogger(__name__)


class DuckDuckGoSearchAgent:
    """
    Agente WEBSEARCH - Trata consultas que requerem informações atualizadas da web usando create_agent e @tool do LangChain.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, verbose: bool = False):
        self.llm = create_chat_llm(model_name=model_name, temperature=temperature)
        self.verbose = verbose
        self.mcp_client = get_mcp_search_client()
        self.callbacks = [StdOutCallbackHandler()] if verbose else []
        
        self.tools = [
            Tool(
                name="web_search",
                func=self._web_search_and_synthesize,
                description="Search the web using DuckDuckGo and synthesize the results into a comprehensive answer. Use for current events, facts, or general knowledge. Input should be a search query string."
            )
        ]
        
        # Criar agente ReAct usando LangChain
        self.agent = create_agent(self.llm, self.tools)

    def _web_search_and_synthesize(self, query: str) -> str:
        """Buscar na web e sintetizar automaticamente os resultados."""
        # Step 1: Get raw search results
        search_results = self._web_search_sync(query)
        
        # Step 2: Automatically synthesize (no agent decision needed)
        synthesized = self._synthesize(search_results)
        
        return synthesized
    
    def _web_search_sync(self, query: str) -> str:
        """Realizar busca web usando cliente MCP."""
        return self.mcp_client.search(query, max_results=5)

    def _synthesize(self, context: str) -> str:
        """Sintetizar informações de resultados de busca e contexto local em uma resposta abrangente."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
              You are a helpful assistant that synthesizes information from web search and local documents.
              
              Your task:
              1. Analyze available information sources
              2. Extract relevant information for the user's query
              3. Provide a clear, accurate answer
              4. Cite sources appropriately
              5. Integrate local knowledge base context with web results when both available
              6. Be transparent about information sources and limitations
              
              Guidelines:
              - If web search results are empty but local context exists, rely on local information
              - If web results exist, mention they come from web search
              - Include source URLs when citing web results
              - Be honest if information is insufficient or outdated
              - Prefer local documents for specific company/project information
              - Prefer web search for general knowledge, current events, or external information
             """),
            ("user", "Context:\n{context}\n\nProvide a comprehensive synthesized answer:")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        return response.content

    def run(self, query: str, rag_context: Optional[str] = None) -> Dict[str, Any]:
        """Executar o agente com a consulta fornecida e contexto RAG opcional."""
        try:
            agent_input = query
            if rag_context:
                agent_input = f"User Question: {query}\n\nLocal Knowledge Base Context:\n{rag_context}\n\nPlease answer the question using web search if needed and synthesize all available information."
            
            # Agente LangGraph retorna um dict com chave 'messages'
            result = self.agent.invoke({"messages": [("user", agent_input)]})
            
            # Extrair o conteúdo da mensagem final
            final_response = result.get("messages", [])[-1].content if result.get("messages") else str(result)
            
            return {
                "response": final_response,
                "num_sources": 1,
                "search_performed": True
            }
        except Exception as e:
            logger.error(f"[DuckDuckGo Agent] Agent error: {e}", exc_info=True)
            fallback = "I couldn't find information in the local knowledge base, and web search is currently unavailable. "
            if rag_context:
                fallback += f"However, here's what I found locally:\n\n{rag_context}"
            else:
                fallback += "Please try rephrasing your question or check if the information exists in the uploaded documents."
            return {
                "response": fallback,
                "num_sources": 0,
                "search_performed": False,
                "error": str(e)
            }


_duckduckgo_agent = None

def get_duckduckgo_agent() -> DuckDuckGoSearchAgent:
    """DuckDuckGo agent instance"""
    global _duckduckgo_agent
    if _duckduckgo_agent is None:
        _duckduckgo_agent = DuckDuckGoSearchAgent()
    return _duckduckgo_agent
