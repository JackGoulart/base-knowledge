"""
Endpoints de gerenciamento de documentos
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging

from shared.database.config import get_db
from shared.repository.document_repository import DocumentRepository
from shared.models.document import DocumentStatus
from schemas.document_schemas import (
    DocumentResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentChunkResponse,
    ChunksListResponse,
    DeleteResponse,
    UpdateDocumentRequest,
    UpdateDocumentResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()



@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Obter documento por ID com todos os metadados"""
    repo = DocumentRepository(db)
    document = repo.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status.value,
        file_size=document.file_size,
        num_chunks=document.num_chunks,
        chunk_size=document.chunk_size,
        embedding_model=document.embedding_model,
        created_at=document.created_at.isoformat(),
        completed_at=document.completed_at.isoformat() if document.completed_at else None,
        markdown_preview=document.markdown_content[:500] + "..." if document.markdown_content and len(document.markdown_content) > 500 else document.markdown_content,
        error_message=document.error_message
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Listar todos os documentos com paginação e filtro de status opcional"""
    repo = DocumentRepository(db)
    
    # Converter string de status para enum se fornecido
    status_filter = None
    if status:
        try:
            status_filter = DocumentStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join([s.value for s in DocumentStatus])}"
            )
    
    documents = repo.get_all_documents(skip=skip, limit=limit, status=status_filter)
    total_count = repo.get_document_count(status=status_filter)
    
    return DocumentListResponse(
        total=total_count,
        skip=skip,
        limit=limit,
        documents=[
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                status=doc.status.value,
                file_size=doc.file_size,
                num_chunks=doc.num_chunks,
                created_at=doc.created_at.isoformat(),
                completed_at=doc.completed_at.isoformat() if doc.completed_at else None
            )
            for doc in documents
        ]
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Excluir documento e todos os chunks associados"""
    repo = DocumentRepository(db)
    
    if not repo.get_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    deleted = repo.delete_document(document_id)
    
    if deleted:
        return DeleteResponse(
            message="Document deleted successfully",
            document_id=document_id
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.put("/{document_id}", response_model=UpdateDocumentResponse)
async def update_document(
    document_id: int,
    update_data: UpdateDocumentRequest,
    db: Session = Depends(get_db)
):
    """Atualizar metadados do documento"""
    repo = DocumentRepository(db)
    
    document = repo.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    updated_fields = {}
    
    # Atualizar filename se fornecido
    if update_data.filename is not None:
        document.filename = update_data.filename
        updated_fields["filename"] = update_data.filename
    
    if updated_fields:
        db.commit()
        db.refresh(document)
        
        return UpdateDocumentResponse(
            message="Document updated successfully",
            document_id=document_id,
            updated_fields=updated_fields
        )
    else:
        return UpdateDocumentResponse(
            message="No fields to update",
            document_id=document_id,
            updated_fields={}
        )


@router.get("/{document_id}/chunks", response_model=ChunksListResponse)
async def get_document_chunks(
    document_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Obter todos os chunks de um documento específico"""
    repo = DocumentRepository(db)
    
    document = repo.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = repo.get_chunks_by_document(document_id, skip=skip, limit=limit)
    total_count = repo.get_chunk_count(document_id=document_id)
    
    return ChunksListResponse(
        document_id=document_id,
        total_chunks=total_count,
        skip=skip,
        limit=limit,
        chunks=[
            DocumentChunkResponse(
                id=chunk.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                text_length=chunk.text_length,
                metadata=chunk.chunk_metadata,
                created_at=chunk.created_at.isoformat()
            )
            for chunk in chunks
        ]
    )
