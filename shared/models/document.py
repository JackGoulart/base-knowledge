"""
Modelos SQLAlchemy Document e DocumentChunk com suporte a pgvector
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import enum
from shared.database.config import Base


class DocumentStatus(str, enum.Enum):
    """Status de processamento do documento"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """
    Modelo de documento representando documentos enviados
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_extension = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False) 
    
    # Status de processamento
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PROCESSING, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Conteúdo do documento
    markdown_content = Column(Text, nullable=True)
    markdown_length = Column(Integer, nullable=True)
    
    # # Metadados de processamento
    num_chunks = Column(Integer, default=0)
    chunk_size = Column(Integer, nullable=False)
    embedding_model = Column(String(100), default="text-embedding-3-small")
    embedding_dimension = Column(Integer, default=1536) # Dimensão dos vetores de embedding
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadados adicionais
    doc_metadata = Column(JSON, default={})
    
    # Relacionamentos
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class DocumentChunk(Base):
    """
    Modelo DocumentChunk representando chunks de texto com embeddings
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Informações do chunk
    chunk_index = Column(Integer, nullable=False)  # Ordem dentro do documento
    text = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)
    
    # Vetor de embedding (usando pgvector)
    embedding = Column(Vector(1536), nullable=True)  # 1536 dimensões para text-embedding-3-small
    
    # Chunk metadata (headings, doc_items, etc.)
    chunk_metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relacionamentos
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


# Índice composto para filtrar por documento e ordenar por chunk_index
Index('idx_chunks_document_chunk_index', DocumentChunk.document_id, DocumentChunk.chunk_index)

# Índice para filtrar documentos por status
Index('idx_documents_status', Document.status)

# Índice para ordenar documentos por data de criação
Index('idx_documents_created_at', Document.created_at.desc())
