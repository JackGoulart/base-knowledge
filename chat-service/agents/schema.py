"""
Centralized schema definitions for agents
Contains TypedDict and Pydantic models used across the agent system
"""
from typing import List, Dict, Any, Optional, Sequence, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class OrchestratorState(TypedDict):
    """State for multi-agent orchestration"""
    query: str
    conversation_history: List[Dict[str, str]]
    selected_agent: str  # Which agent to use
    response: str
    num_sources: int
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    """State passed between agent nodes in RAG workflow"""
    messages: Sequence[BaseMessage]
    query: str
    context: List[Dict[str, Any]]
    analysis: str
    final_response: str
    next_action: str

class DocumentInfo(BaseModel):
    """Information about a single document"""
    id: str = Field(description="Document ID")
    filename: str = Field(description="Document filename")
    status: str = Field(description="Processing status")
    file_size: int = Field(description="File size in bytes")
    num_chunks: int = Field(description="Number of chunks")
    created_at: str = Field(description="Upload timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")


class DocumentListResponse(BaseModel):
    """Structured response for document listing queries"""
    response: str = Field(description="Natural language response")
    documents: List[DocumentInfo] = Field(default_factory=list, description="List of documents")
    total_documents: int = Field(description="Total number of documents")
    error: Optional[str] = Field(default=None, description="Error message if any")
