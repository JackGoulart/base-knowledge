"""
Agentes de IA usando LangChain e LangGraph para análise de documentos e RAG
"""
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging

from shared.repository.document_repository import DocumentRepository
from shared.database.config import SessionLocal
from services.openai_client import create_chat_llm, create_embeddings
from agents.schema import AgentState

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Coordenador principal de agentes usando LangGraph"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = create_chat_llm(model_name=model_name, temperature=temperature)
        self.embeddings = create_embeddings()
        self.graph = self._create_graph()
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Nó RAG: Recuperar chunks de documentos relevantes do banco de dados vetorial"""
        logger.info(f"[RAG Agent] Retrieving context for query: {state['query']}")
        
        try:
            # Gerar embedding para a consulta
            query_embedding = self.embeddings.embed_query(state["query"])
            
            # Buscar no banco de dados vetorial
            db = SessionLocal()
            repo = DocumentRepository(db)
            
            results = repo.search_similar_chunks(
                embedding=query_embedding,
                limit=5,
                document_id=None
            )
            
            # Formatar contexto
            context = []
            for chunk, distance in results:
                context.append({
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "similarity": 1 - float(distance),
                    "metadata": chunk.chunk_metadata
                })
            
            db.close()
            
            logger.info(f"[RAG Agent] Retrieved {len(context)} relevant chunks")
            
            # Atualizar estado
            state["context"] = context
            state["next_action"] = "analyze"
            
            return state
            
        except Exception as e:
            logger.error(f"[RAG Agent] Error retrieving context: {str(e)}", exc_info=True)
            state["context"] = []
            state["next_action"] = "respond"
            return state
    
    def _analyze_context(self, state: AgentState) -> AgentState:
        """Nó de Análise: Analisar contexto recuperado e extrair insights"""
        logger.info("[Analysis Agent] Analyzing retrieved context")
        
        if not state["context"]:
            logger.warning("[Analysis Agent] No context to analyze")
            state["analysis"] = "No relevant documents found in the knowledge base."
            state["next_action"] = "respond"
            return state
        
        try:
            # Preparar texto do contexto
            context_text = "\n\n".join([
                f"[Document {ctx['document_id']}, Chunk {ctx['chunk_index']}]:\n{ctx['text']}"
                for ctx in state["context"]
            ])
            
            # Prompt de análise
            analysis_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert document analyst. Your task is to:
                1. Analyze the provided document chunks
                2. Identify key information, patterns, and insights
                3. Summarize the most relevant points related to the user's query
                4. Note any contradictions or gaps in information

                Be concise but thorough in your analysis."""),
                                HumanMessage(content=f"""Query: {state["query"]}

                Retrieved Context:
                {context_text}

                Provide your analysis:""")
            ])
            
            # Get analysis from LLM
            response = self.llm.invoke(analysis_prompt.format_messages())
            analysis = response.content
            
            logger.info("[Analysis Agent] Analysis completed")
            
            # Update state
            state["analysis"] = analysis
            state["next_action"] = "respond"
            
            return state
            
        except Exception as e:
            logger.error(f"[Analysis Agent] Error during analysis: {str(e)}", exc_info=True)
            state["analysis"] = "Error occurred during analysis."
            state["next_action"] = "respond"
            return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Response Node: Generate final response based on analysis"""
        logger.info("[Response Agent] Generating final response")
        
        try:
            # Prepare response prompt
            response_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a helpful AI assistant with access to a document knowledge base. 
                Your task is to provide accurate, helpful responses based on the analyzed context.
                
                Guidelines:
                - Answer the user's query directly and clearly
                - Cite specific document sections when relevant
                - If the information is incomplete, acknowledge it
                - Maintain a conversational and helpful tone
                - If no relevant context was found, politely indicate that"""),
                                HumanMessage(content=f"""User Query: {state["query"]}
                
                Analysis of Retrieved Documents:
                {state["analysis"]}
                
                Retrieved Context Summary:
                {len(state["context"])} relevant chunks found with similarities ranging from {min([c['similarity'] for c in state['context']], default=0):.2f} to {max([c['similarity'] for c in state['context']], default=0):.2f}
                Generate a helpful response to the user:""")
            ])
            
            # Get final response from LLM
            response = self.llm.invoke(response_prompt.format_messages())
            final_response = response.content
            
            logger.info("[Response Agent] Response generated successfully")
            
            # Update state
            state["final_response"] = final_response
            state["next_action"] = "end"
            
            return state
            
        except Exception as e:
            logger.error(f"[Response Agent] Error generating response: {str(e)}", exc_info=True)
            state["final_response"] = "I apologize, but I encountered an error while generating a response. Please try again."
            state["next_action"] = "end"
            return state
    
    def _route_next(self, state: AgentState) -> str:
        """Router: Determine next node based on state"""
        next_action = state.get("next_action", "retrieve")
        
        if next_action == "analyze":
            return "analyze"
        elif next_action == "respond":
            return "respond"
        else:
            return END
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("analyze", self._analyze_context)
        workflow.add_node("respond", self._generate_response)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        workflow.add_conditional_edges(
            "retrieve",
            self._route_next,
            {
                "analyze": "analyze",
                "respond": "respond",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "analyze",
            self._route_next,
            {
                "respond": "respond",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "respond",
            self._route_next,
            {
                END: END
            }
        )
        
        return workflow.compile()
    
    def chat(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a chat query through the agent graph
        
        Args:
            query: User's question/query
            conversation_history: Optional list of previous messages
            
        Returns:
            Dict with response and metadata
        """
        logger.info(f"[Agent] Processing query: {query}")
        
        # Initialize state
        messages = []
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        initial_state = AgentState(
            messages=messages,
            query=query,
            context=[],
            analysis="",
            final_response="",
            next_action="retrieve"
        )
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Prepare response
            response = {
                "response": final_state["final_response"],
                "num_sources": len(final_state["context"])
            }
            
            logger.info(f"[Agent] Query processed successfully with {response['num_sources']} sources")
            return response
            
        except Exception as e:
            logger.error(f"[Agent] Error processing query: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "num_sources": 0,
                "error": str(e)
            }


_agent_instance = None

def get_agent() -> DocumentAgent:
    """singleton instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DocumentAgent()
    return _agent_instance
