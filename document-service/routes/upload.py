"""
Endpoint de upload de documentos
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pathlib import Path
import tempfile
import os
import uuid
import logging
from datetime import datetime

from shared.database.config import get_db
from shared.repository.document_repository import DocumentRepository
from services.document_processor import process_document_task

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = 512,
    db: Session = Depends(get_db)
):
    """
    Upload de documento PDF ou DOCX para processamento.
    
    O pipeline irá:
    1. Converter o documento para Markdown
    2. Realizar chunking híbrido usando pipeline otimizado do Docling
    3. Gerar embeddings para cada chunk
    4. Salvar no PostgreSQL Vector DB
    
    Args:
        file: Arquivo de documento enviado (PDF ou DOCX)
        chunk_size: Tamanho alvo para chunks em tokens (padrão: 512)
        
    Returns:
        Job ID e Document ID para rastreamento
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file"
        )
    
    # Create temporary file to store upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Received file: {file.filename}")
        
        # Create document record in database
        repo = DocumentRepository(db)
        document = repo.create_document(
            filename=file.filename,
            file_extension=file_extension,
            file_size=len(content),
            chunk_size=chunk_size
        )
        
        # Create job ID for background task tracking
        job_id = str(uuid.uuid4())
        
        # Get jobs_storage from app state
        jobs_storage = request.app.state.jobs_storage
        
        # Initialize job tracking
        jobs_storage[job_id] = {
            "job_id": job_id,
            "document_id": document.id,
            "status": "processing",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "result": None
        }
        
        # Add background task
        background_tasks.add_task(
            process_document_task,
            job_id=job_id,
            document_id=document.id,
            tmp_file_path=tmp_file_path,
            filename=file.filename,
            chunk_size=chunk_size,
            jobs_storage=jobs_storage
        )
        
        return JSONResponse(content={
            "job_id": job_id,
            "document_id": document.id,
            "status": "processing",
            "message": "Document processing started in background",
            "check_status_url": f"/jobs/{job_id}",
            "document_url": f"/documents/{document.id}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
