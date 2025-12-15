"""
Conversation management endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel

from shared.database.config import get_db
from shared.repository.conversation_repository import ConversationRepository

router = APIRouter()


class ConversationResponse(BaseModel):
    id: int
    session_id: str
    title: Optional[str]
    created_at: str
    updated_at: Optional[str]


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources_count: int
    created_at: str


@router.get("/{session_id}")
async def get_conversation(session_id: str, db: Session = Depends(get_db)):
    """Get conversation by session ID"""
    repo = ConversationRepository(db)
    conversation = repo.get_conversation_by_session(session_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationResponse(
        id=conversation.id,
        session_id=conversation.session_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None
    )


@router.get("/{session_id}/messages")
async def get_conversation_messages(
    session_id: str,
    limit: Optional[int] = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get messages for a conversation"""
    repo = ConversationRepository(db)
    conversation = repo.get_conversation_by_session(session_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = repo.get_conversation_messages(conversation.id, limit=limit)
    
    return {
        "session_id": session_id,
        "total_messages": len(messages),
        "messages": [
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources_count=msg.sources_count,
                created_at=msg.created_at.isoformat()
            )
            for msg in messages
        ]
    }
