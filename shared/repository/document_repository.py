"""
Padrão Repository para operações CRUD de Documentos
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from datetime import datetime
import logging

from shared.models.document import Document, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository para operações CRUD de Document e DocumentChunk
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ============= Operações CRUD de Documentos =============
    
    def create_document(
        self,
        filename: str,
        file_extension: str,
        file_size: int,
        chunk_size: int,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Criar um novo registro de documento"""
        logger.info(f"Creating document record: {filename} ({file_size} bytes)")
        document = Document(
            filename=filename,
            file_extension=file_extension,
            file_size=file_size,
            chunk_size=chunk_size,
            status=DocumentStatus.PROCESSING,
            doc_metadata=doc_metadata or {}
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        logger.info(f"Document created with ID: {document.id}")
        return document
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Obter documento por ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_document_with_chunks(self, document_id: int) -> Optional[Document]:
        """Get document by ID with all chunks loaded"""
        return self.db.query(Document).filter(
            Document.id == document_id
        ).first()
    
    def get_all_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[DocumentStatus] = None
    ) -> List[Document]:
        """Get all documents with pagination and optional status filter"""
        query = self.db.query(Document)
        
        if status:
            query = query.filter(Document.status == status)
        
        return query.order_by(desc(Document.created_at)).offset(skip).limit(limit).all()
    
    def update_document_status(
        self,
        document_id: int,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Update document processing status"""
        logger.info(f"Updating document {document_id} status to {status.value}")
        document = self.get_document(document_id)
        if document:
            document.status = status
            document.error_message = error_message
            
            if status == DocumentStatus.COMPLETED:
                document.completed_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(document)
            logger.info(f"Document {document_id} status updated successfully")
        else:
            logger.warning(f"Document {document_id} not found for status update")
        return document
    
    def update_document_content(
        self,
        document_id: int,
        markdown_content: str,
        markdown_length: int,
        num_chunks: int
    ) -> Optional[Document]:
        """Update document content after processing"""
        logger.debug(f"Updating content for document {document_id}: {markdown_length} chars, {num_chunks} chunks")
        document = self.get_document(document_id)
        if document:
            document.markdown_content = markdown_content
            document.markdown_length = markdown_length
            document.num_chunks = num_chunks
            
            self.db.commit()
            self.db.refresh(document)
            logger.info(f"Document {document_id} content updated")
        else:
            logger.warning(f"Document {document_id} not found for content update")
        return document
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all associated chunks (cascade)"""
        logger.info(f"Deleting document {document_id}")
        document = self.get_document(document_id)
        if document:
            self.db.delete(document)
            self.db.commit()
            logger.info(f"Document {document_id} and associated chunks deleted successfully")
            return True
        logger.warning(f"Document {document_id} not found for deletion")
        return False
    
    # ============= DocumentChunk CRUD Operations =============
    
    def create_chunk(
        self,
        document_id: int,
        chunk_index: int,
        text: str,
        embedding: List[float],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """Create a new document chunk with embedding"""
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            text_length=len(text),
            embedding=embedding,
            chunk_metadata=chunk_metadata or {}
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk
    
    def create_chunks_batch(
        self,
        document_id: int,
        chunks_data: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Create multiple chunks in batch"""
        logger.info(f"Creating batch of {len(chunks_data)} chunks for document {document_id}")
        chunks = []
        for data in chunks_data:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=data["chunk_index"],
                text=data["text"],
                text_length=len(data["text"]),
                embedding=data.get("embedding"),
                chunk_metadata=data.get("chunk_metadata", {})
            )
            chunks.append(chunk)
        
        logger.debug(f"Bulk saving {len(chunks)} chunks...")
        self.db.bulk_save_objects(chunks, return_defaults=True)
        self.db.commit()
        logger.info(f"Successfully saved {len(chunks)} chunks for document {document_id}")
        return chunks
    
    def get_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Get chunk by ID"""
        return self.db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
    
    def get_chunks_by_document(
        self,
        document_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(asc(DocumentChunk.chunk_index)).offset(skip).limit(limit).all()
    
    def search_similar_chunks(
        self,
        embedding: List[float],
        limit: int = 10,
        document_id: Optional[int] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Buscar chunks similares usando similaridade vetorial (distância cosseno).
        Usa automaticamente o índice HNSW se disponível para performance.
        
        Returns:
            Lista de tuplas (chunk, distance) ordenada por similaridade
        """
        scope = f"document {document_id}" if document_id else "all documents"
        logger.info(f"Searching for {limit} similar chunks in {scope}")
        
        # Esta query usa automaticamente idx_document_chunks_embedding_hnsw
        # para busca vetorial rápida via operador <=> (cosine_distance)
        query = self.db.query(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(embedding).label("distance")
        )
        
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        
        results = query.order_by(asc("distance")).limit(limit).all()
        logger.info(f"Found {len(results)} similar chunks")
        return results
    
    def delete_chunk(self, chunk_id: int) -> bool:
        """Delete a specific chunk"""
        chunk = self.get_chunk(chunk_id)
        if chunk:
            self.db.delete(chunk)
            self.db.commit()
            return True
        return False
    
    def delete_chunks_by_document(self, document_id: int) -> int:
        """Delete all chunks for a document. Returns count of deleted chunks."""
        count = self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete()
        self.db.commit()
        return count
    
    def get_document_count(self, status: Optional[DocumentStatus] = None) -> int:
       """Get total count of documents"""
       query = self.db.query(Document)
       if status:
           query = query.filter(Document.status == status)
       return query.count()
    
    def get_chunk_count(self, document_id: Optional[int] = None) -> int:
        """Get total count of chunks"""
        query = self.db.query(DocumentChunk)
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        return query.count()