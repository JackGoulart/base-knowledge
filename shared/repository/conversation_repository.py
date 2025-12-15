"""
Repository para operações CRUD de Conversação e Mensagem
"""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
import uuid
import logging

from shared.models.conversation import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository para operações de Conversação e Mensagem
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ============= Operações de Conversação =============
    
    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """Criar uma nova conversação com session_id único"""
        session_id = str(uuid.uuid4())
        logger.info(f"Creating conversation with session_id: {session_id}")
        
        conversation = Conversation(
            session_id=session_id,
            title=title
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        
        logger.info(f"Conversation created with ID: {conversation.id}")
        return conversation
    
    def get_conversation_by_session(self, session_id: str) -> Optional[Conversation]:
        """Obter conversação por session_id"""
        return self.db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).first()
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Obter conversação por ID"""
        return self.db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
    
    def get_or_create_conversation(self, session_id: Optional[str] = None) -> Conversation:
        """Get existing conversation or create new one if session_id not found"""
        if session_id:
            conversation = self.get_conversation_by_session(session_id)
            if conversation:
                logger.info(f"Using existing conversation: {session_id}")
                return conversation
        
        logger.info("Creating new conversation")
        return self.create_conversation()
    
    # ============= Message Operations =============
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources_count: int = 0
    ) -> Message:
        """Add a message to a conversation"""
        logger.info(f"Adding {role} message to conversation {conversation_id}")
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_count=sources_count
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        return message
    
    def get_conversation_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get all messages for a conversation, ordered by creation time"""
        query = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = 20
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for agent
        Returns list of dicts with 'role' and 'content'
        """
        conversation = self.get_conversation_by_session(session_id)
        if not conversation:
            return []
        
        messages = self.get_conversation_messages(conversation.id, limit=limit)
        
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
