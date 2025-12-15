"""
Endpoints de chat
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List
import logging
import uuid
import gradio as gr

from shared.database.config import get_db
from shared.repository.conversation_repository import ConversationRepository
from agents.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    response: str
    session_id: str
    num_sources: int


@router.post("/qa", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Endpoint de chat com agentes de IA usando RAG e análise.
    Usa PostgreSQL para persistir histórico de conversas.
    
    O fluxo dos agentes:
    1. Router: Seleciona o agente apropriado (RAG, lista de documentos ou busca web)
    2. Agente Selecionado: Processa a consulta com lógica especializada
    3. Retorna resposta simples com contagem de fontes
    
    Args:
        query: Pergunta do usuário
        session_id: ID de sessão opcional para continuidade da conversa (gerado automaticamente se não fornecido)
        
    Returns:
        Resposta gerada por IA com session_id e contagem de fontes
    """
    logger.info(f"[Chat] Received query: {request.query}")
    
    try:
        # Get or create conversation
        conv_repo = ConversationRepository(db)
        conversation = conv_repo.get_or_create_conversation(session_id=request.session_id)
        
        # Get conversation history from database
        history = conv_repo.get_conversation_history(conversation.session_id, limit=20)
        logger.info(f"[Chat] Loaded {len(history)} messages from conversation {conversation.session_id}")
        
        # Save user message
        conv_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.query
        )
        
        # Get orchestrator and process query
        orchestrator = get_orchestrator()
        result = orchestrator.process_query(
            query=request.query,
            conversation_history=history
        )
        
        # Save assistant response
        conv_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=result["response"],
            sources_count=result.get("num_sources", 0)
        )
        
        # Prepare response
        response = ChatResponse(
            response=result["response"],
            session_id=conversation.session_id,
            num_sources=result.get("num_sources", 0)
        )
        
        logger.info(f"[Chat] Response generated with {response.num_sources} sources")
        return response
        
    except Exception as e:
        logger.error(f"[Chat] Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


def gradio_chat_function(message: str, history: List[List[str]]):
    """Função de chat para interface Gradio"""
    try:
        session_id = "gradio-session"
        
        request = ChatRequest(
            query=message,
            session_id=session_id
        )
        
        from shared.database.config import SessionLocal
        db = SessionLocal()
        
        try:
            conv_repo = ConversationRepository(db)
            conversation = conv_repo.get_or_create_conversation(session_id=request.session_id)
            
            # Save user message
            conv_repo.add_message(
                conversation_id=conversation.id,
                role="user",
                content=request.query
            )
            
            # Get conversation history from database
            history_msgs = conv_repo.get_conversation_history(conversation.session_id, limit=20)
            
            # Get orchestrator and process query
            orchestrator = get_orchestrator()
            result = orchestrator.process_query(
                query=request.query,
                conversation_history=history_msgs
            )
            
            # Save assistant response
            conv_repo.add_message(
                conversation_id=conversation.id,
                role="assistant",
                content=result["response"],
                sources_count=result.get("num_sources", 0)
            )
            
            response_text = result["response"]
            
            if result.get("num_sources", 0) > 0:
                response_text += f"\n\n*[Based on {result['num_sources']} source(s)]*"
            
            return response_text
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"[Gradio] Error: {str(e)}", exc_info=True)
        return f"Sorry, an error occurred: {str(e)}"


def create_gradio_interface():
    """Criar e retornar interface de chat Gradio"""
    return gr.ChatInterface(
        fn=gradio_chat_function,
        title="Base Knowledge Chat",
        description="Ask questions about your documents using AI-powered retrieval and analysis.",
        examples=[
            "What are the main topics in the documents?",
            "Summarize the key points",
            "Tell me about..."
        ]
    )
