"""
Orquestrador de Múltiplos Agentes usando LangGraph
Coordenar entre agentes especializados usando roteamento baseado em LLM
"""
from typing import Dict, Any, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import logging

from agents.document_agent import DocumentAgent
from agents.document_list_agent import DocumentListAgent
from agents.duckduckgo_agent import DuckDuckGoSearchAgent
from services.openai_client import create_chat_llm
from agents.schema import OrchestratorState

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orquestra múltiplos agentes especializados usando roteamento baseado em LLM
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.router_llm = create_chat_llm(model_name=model_name, temperature=0)
        
        # Inicializar agentes especializados
        self.rag_agent = DocumentAgent(model_name=model_name)
        self.doc_list_agent = DocumentListAgent(model_name=model_name)
        self.web_search_agent = DuckDuckGoSearchAgent(model_name=model_name)
        
        # Criar grafo de orquestração
        self.graph = self._create_graph()
    
    def _route_query(self, state: OrchestratorState) -> OrchestratorState:
        """
        Usar LLM para rotear inteligentemente a consulta para o agente apropriado
        """
        logger.info("[Orchestrator] Routing query to appropriate agent")
        
        routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent query router for a document knowledge base system.
                Your task is to analyze the user's query and determine which specialized agent should handle it.
                
                Available Agents:
                1. DOCUMENT_LIST - Handles queries about:
                   - What documents are available/uploaded
                   - Document metadata (status, size, count, names)
                   - Listing files in the database
                   - Document upload times and completion status
                   
                2. RAG - Handles queries about:
                   - Content within documents
                   - Questions that need semantic search
                   - Analysis of document content
                   - Information retrieval from document text
                   - Summarization and insights from documents
                
                3. WEBSEARCH - Handles queries that require up-to-date web information:
                     - Current events and breaking news
                     - Recent information or facts not in documents
                     - General knowledge queries explicitly asking for web search
                          
                Rules:
                - If asking ABOUT documents (metadata), use DOCUMENT_LIST
                - If asking WHAT'S IN documents (content), use RAG
                - If explicitly asking for current/web info, use WEBSEARCH
                - Default to RAG for ambiguous queries
                - Note: RAG will automatically fall back to WEBSEARCH if no local results found
                
                Respond with ONLY the agent name: DOCUMENT_LIST, RAG, or WEBSEARCH"""),
            HumanMessage(content=f"User Query: {state['query']}\n\nWhich agent should handle this?")
        ])
        
        try:
            response = self.router_llm.invoke(routing_prompt.format_messages())
            selected_agent = response.content.strip().upper()
            
            # Validar resposta
            if selected_agent not in ["DOCUMENT_LIST", "RAG", "WEBSEARCH"]:
                logger.warning(f"[Orchestrator] Invalid agent selection '{selected_agent}', defaulting to RAG")
                selected_agent = "RAG"
            
            state["selected_agent"] = selected_agent
            logger.info(f"[Orchestrator] Routed to {selected_agent} agent")
            
        except Exception as e:
            logger.error(f"[Orchestrator] Routing error: {str(e)}, defaulting to RAG", exc_info=True)
            state["selected_agent"] = "RAG"
        
        return state
    
    def _execute_document_list_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Executar agente de lista de documentos"""
        logger.info("[Orchestrator] Executing Document List Agent")
        
        try:
            result = self.doc_list_agent.handle_query(state["query"])
            
            # result agora é um modelo Pydantic DocumentListResponse
            state["response"] = result.response
            state["num_sources"] = 0
            state["metadata"] = {
                "agent": "document_list"
            }
            
            logger.info("[Orchestrator] Document List Agent completed")
            
        except Exception as e:
            logger.error(f"[Orchestrator] Document List Agent error: {str(e)}", exc_info=True)
            state["response"] = f"Error retrieving document list: {str(e)}"
            state["num_sources"] = 0
            state["metadata"] = {"agent": "document_list", "error": str(e)}
        
        return state
    
    def _execute_rag_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Executar agente RAG"""
        logger.info("[Orchestrator] Executing RAG Agent")
        
        try:
            result = self.rag_agent.chat(
                query=state["query"],
                conversation_history=state.get("conversation_history", [])
            )
            
            state["response"] = result["response"]
            state["num_sources"] = result.get("num_sources", 0)
            state["metadata"] = {
                "agent": "rag"
            }
            
            logger.info(f"[Orchestrator] RAG Agent completed with {state['num_sources']} sources")
            
        except Exception as e:
            logger.error(f"[Orchestrator] RAG Agent error: {str(e)}", exc_info=True)
            state["response"] = f"Error processing query: {str(e)}"
            state["num_sources"] = 0
            state["metadata"] = {"agent": "rag", "error": str(e)}
        
        return state
    
    def _execute_web_search_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Executar agente de busca web diretamente (quando explicitamente solicitado)"""
        logger.info("[Orchestrator] Executing Web Search Agent (direct request)")
        
        try:
            result = self.web_search_agent.run(
                query=state["query"],
                rag_context=None
            )
            
            # result agora é um dict com resposta simplificada
            state["response"] = result["response"]
            state["num_sources"] = result["num_sources"]
            state["metadata"] = {
                "agent": "web_search"
            }
            
            logger.info(f"[Orchestrator] Web Search Agent completed with {state['num_sources']} sources")
            
        except Exception as e:
            logger.error(f"[Orchestrator] Web Search Agent error: {str(e)}", exc_info=True)
            state["response"] = f"Error performing web search: {str(e)}"
            state["num_sources"] = 0
            state["metadata"] = {"agent": "web_search", "error": str(e)}
        
        return state
    

    def _route_to_agent(self, state: OrchestratorState) -> Literal["document_list", "rag", "web_search"]:
        """Função de roteador para determinar próximo nó"""
        if state["selected_agent"] == "DOCUMENT_LIST":
            return "document_list"
        elif state["selected_agent"] == "RAG":
            return "rag"
        else:
            return "web_search"
    
    def _create_graph(self) -> StateGraph:
        """Criar o grafo de orquestração"""
        workflow = StateGraph(OrchestratorState)
        
        # Adicionar nós
        workflow.add_node("route", self._route_query)
        workflow.add_node("document_list", self._execute_document_list_agent)
        workflow.add_node("rag", self._execute_rag_agent)
        workflow.add_node("web_search", self._execute_web_search_agent)
        
        # Definir ponto de entrada
        workflow.set_entry_point("route")
        
        # Adicionar arestas condicionais do roteador
        workflow.add_conditional_edges(
            "route",
            self._route_to_agent,
            {
                "document_list": "document_list",
                "rag": "rag",
                "web_search": "web_search"
            }
        )
        
        # Todos os agentes levam ao END
        workflow.add_edge("document_list", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("web_search", END)
        
        return workflow.compile()
    
    def process_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Processar consulta através do sistema de agentes orquestrados
        
        Args:
            query: Pergunta do usuário
            conversation_history: Contexto de conversa opcional
            
        Returns:
            Dict com resposta e metadados
        """
        logger.info(f"[Orchestrator] Processing query: {query}")
        
        initial_state = OrchestratorState(
            query=query,
            conversation_history=conversation_history or [],
            selected_agent="",
            response="",
            num_sources=0,
            metadata={}
        )
        
        try:
            # Executar grafo de orquestração
            final_state = self.graph.invoke(initial_state)
            
            result = {
                "response": final_state["response"],
                "num_sources": final_state["num_sources"],
                "agent_used": final_state["metadata"].get("agent")
            }
            
            logger.info(f"[Orchestrator] Query processed by {result['agent_used']} agent")
            return result
            
        except Exception as e:
            logger.error(f"[Orchestrator] Error processing query: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "num_sources": 0,
                "agent_used": "error"
            }


_orchestrator_instance = None

def get_orchestrator() -> AgentOrchestrator:
    """instância singleton do orquestrador"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance