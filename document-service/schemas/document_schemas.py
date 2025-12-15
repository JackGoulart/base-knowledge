"""
Schemas relacionados a documentos
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Schema de resposta para um documento individual"""
    id: int
    filename: str
    status: str
    file_size: int
    num_chunks: int
    chunk_size: int
    embedding_model: str
    created_at: str
    completed_at: Optional[str] = None
    markdown_preview: Optional[str] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListItem(BaseModel):
    """Schema para item na lista de documentos"""
    id: int
    filename: str
    status: str
    file_size: int
    num_chunks: int
    created_at: str
    completed_at: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema de resposta para listagem de documentos"""
    total: int
    skip: int
    limit: int
    documents: List[DocumentListItem]


class DocumentChunkResponse(BaseModel):
    """Schema de resposta para um chunk de documento"""
    id: int
    chunk_index: int
    text: str
    text_length: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    
    class Config:
        from_attributes = True


class ChunksListResponse(BaseModel):
    """Schema de resposta para listagem de chunks"""
    document_id: int
    total_chunks: int
    skip: int
    limit: int
    chunks: List[DocumentChunkResponse]


class DeleteResponse(BaseModel):
    """Schema de resposta para exclusão de documento"""
    message: str
    document_id: int


class UpdateDocumentRequest(BaseModel):
    """Schema de requisição para atualização de documento"""
    filename: Optional[str] = None
    
    class Config:
        from_attributes = True


class UpdateDocumentResponse(BaseModel):
    """Schema de resposta para atualização de documento"""
    message: str
    document_id: int
    updated_fields: Dict[str, Any]

