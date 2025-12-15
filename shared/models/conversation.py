"""
Modelos Conversation e Message para persistência de histórico de chat
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from shared.database.config import Base


class Conversation(Base):
    """
    Modelo Conversation/Session para rastreamento de sessões de chat
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=True)  # Título opcional da conversação
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relacionamentos
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation(id={self.id}, session_id='{self.session_id}')>"


class Message(Base):
    """
    Modelo Message para armazenar mensagens individuais de chat
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Conteúdo da mensagem
    role = Column(String(20), nullable=False)  # 'user' ou 'assistant'
    content = Column(Text, nullable=False)
    
    # Metadados opcionais sobre fontes utilizadas para esta resposta
    sources_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relacionamentos
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role='{self.role}')>"


# Criar índices para melhor performance de consultas
Index('idx_messages_conversation_created', Message.conversation_id, Message.created_at)
