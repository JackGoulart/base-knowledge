"""
Schemas Pydantic para o serviço de documentos
"""
from .document_schemas import (
    DocumentResponse,
    DocumentListResponse,
    DocumentChunkResponse,
    ChunksListResponse,
    DeleteResponse,
    UpdateDocumentRequest,
    UpdateDocumentResponse
)

__all__ = [
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentChunkResponse",
    "ChunksListResponse",
    "DeleteResponse",
    "UpdateDocumentRequest",
    "UpdateDocumentResponse"
]
