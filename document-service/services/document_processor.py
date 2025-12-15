"""
Processamento de documentos com Docling e embeddings do OpenAI
"""
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.chunking import HybridChunker
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

from shared.database.config import SessionLocal
from shared.repository.document_repository import DocumentRepository
from shared.models.document import DocumentStatus

logger = logging.getLogger(__name__)

# Configuração
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def _is_azure() -> bool:
    """Verificar se Azure OpenAI deve ser usado baseado nas variáveis de ambiente."""
    return os.getenv("AZURE_OPENAI_ENDPOINT") is not None


def _create_embeddings_client() -> OpenAIEmbeddings:
    """
    Criar uma instância OpenAIEmbeddings ou AzureOpenAIEmbeddings baseado na configuração.
    """
    if _is_azure():
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )


class ChunkPreview(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingData(BaseModel):
    chunk_id: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float]
    embedding_dimension: int


class DocumentProcessingResponse(BaseModel):
    status: str
    filename: str
    markdown_length: int
    num_chunks: int
    max_tokens_per_chunk: int
    embedding_model: str
    embedding_dimension: int
    chunking_method: str
    tokenizer: str
    markdown_preview: str
    chunks_preview: List[ChunkPreview]
    embeddings: List[EmbeddingData]


def initialize_docling_converter() -> DocumentConverter:
    """Inicializar o conversor de documentos Docling com opções otimizadas para pipeline PDF."""
    accelerator_options = AcceleratorOptions(
        num_threads=4,
        device="cpu"
    )
    
    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator_options
    )
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter


def convert_document(file_path: str):
    """Converter documento PDF ou DOCX usando Docling."""
    try:
        logger.info(f"Starting document conversion for: {file_path}")
        converter = initialize_docling_converter()
        logger.debug("Docling converter initialized")
        result = converter.convert(file_path)
        logger.info(f"Successfully converted {file_path}")
        return result.document
    except Exception as e:
        logger.error(f"Error converting document {file_path}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {str(e)}")


def hybrid_chunk_document(doc, max_tokens: int = 512) -> List[Dict[str, Any]]:
    """Realizar chunking híbrido no documento usando HybridChunker do Docling."""
    try:
        logger.info(f"Starting hybrid chunking with max_tokens={max_tokens}")
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=max_tokens,
        )
        logger.debug("HybridChunker initialized")
        
        chunk_iter = chunker.chunk(dl_doc=doc)
        
        chunks = []
        for idx, chunk in enumerate(chunk_iter):
            chunk_data = {
                "text": chunk.text,
                "meta": {
                    "doc_items": [item.self_ref for item in chunk.meta.doc_items] if chunk.meta.doc_items else [],
                    "headings": chunk.meta.headings if hasattr(chunk.meta, 'headings') else []
                }
            }
            chunks.append(chunk_data)
            if idx % 10 == 0:
                logger.debug(f"Processed {idx + 1} chunks...")
        
        logger.info(f"Created {len(chunks)} chunks from document using HybridChunker")
        return chunks
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text chunking failed: {str(e)}")


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gerar embeddings para chunks de texto usando OpenAI ou Azure OpenAI."""
    try:
        # Criar cliente de embeddings (detecta automaticamente OpenAI ou Azure)
        embeddings_client = _create_embeddings_client()
        
        # Extrair todos os textos dos chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Gerar embeddings em batch
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks in batch")
        embedding_vectors = embeddings_client.embed_documents(chunk_texts)
        
        # Montar dados de embeddings com metadados
        embeddings_data = []
        for idx, (chunk, embedding_vector) in enumerate(zip(chunks, embedding_vectors)):
            embeddings_data.append({
                "chunk_id": idx,
                "text": chunk["text"],
                "metadata": chunk.get("meta", {}),
                "embedding": embedding_vector,
                "embedding_dimension": len(embedding_vector)
            })
            
        logger.info(f"Generated embeddings for {len(chunks)} chunks using batch API")
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


def process_document_task(
    job_id: str,
    document_id: int,
    tmp_file_path: str,
    filename: str,
    chunk_size: int,
    jobs_storage: Dict[str, Any]
):
    """Tarefa em segundo plano para processar documentos e salvar no banco de dados."""
    db = SessionLocal()
    repo = DocumentRepository(db)
    
    try:
        logger.info(f"[Job {job_id}] Starting processing for '{filename}' (document_id={document_id})")
        
        # Passo 1: Converter documento usando Docling
        logger.info(f"[Job {job_id}] Step 1/5: Converting document to Docling format")
        doc = convert_document(tmp_file_path)
        
        # Exportar para Markdown para visualização
        logger.info(f"[Job {job_id}] Step 2/5: Exporting to Markdown")
        markdown_content = doc.export_to_markdown()
        logger.info(f"[Job {job_id}] Markdown content length: {len(markdown_content)} characters")
        
        # Passo 2: Realizar chunking híbrido
        logger.info(f"[Job {job_id}] Step 3/5: Performing hybrid chunking (max_tokens={chunk_size})")
        chunks = hybrid_chunk_document(doc, max_tokens=chunk_size)
        
        # Passo 3: Gerar embeddings
        logger.info(f"[Job {job_id}] Step 4/5: Generating embeddings for {len(chunks)} chunks")
        embeddings_data = generate_embeddings(chunks)
        
        # Passo 4: Salvar no banco de dados
        logger.info(f"[Job {job_id}] Step 5/5: Saving to database")
        repo.update_document_content(
            document_id=document_id,
            markdown_content=markdown_content,
            markdown_length=len(markdown_content),
            num_chunks=len(chunks)
        )
        
        # Salvar chunks com embeddings em lote
        chunks_data = [
            {
                "chunk_index": data["chunk_id"],
                "text": data["text"],
                "embedding": data["embedding"],
                "chunk_metadata": data.get("metadata", {})
            }
            for data in embeddings_data
        ]
        logger.info(f"[Job {job_id}] Saving {len(chunks_data)} chunks to database")
        repo.create_chunks_batch(document_id, chunks_data)
        logger.info(f"[Job {job_id}] All chunks saved successfully")
        
        # Atualizar status do documento para concluído
        repo.update_document_status(document_id, DocumentStatus.COMPLETED)
        
        # Limpar arquivo temporário
        os.unlink(tmp_file_path)
        
        # Preparar resposta
        result = DocumentProcessingResponse(
            status="success",
            filename=filename,
            markdown_length=len(markdown_content),
            num_chunks=len(chunks),
            max_tokens_per_chunk=chunk_size,
            embedding_model=EMBEDDING_MODEL,
            embedding_dimension=embeddings_data[0]["embedding_dimension"] if embeddings_data else EMBEDDING_DIMENSION,
            chunking_method="HybridChunker",
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            markdown_preview=markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
            chunks_preview=[
                ChunkPreview(
                    text=c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                    metadata=c.get("meta", {})
                ) for c in chunks[:2]
            ],
            embeddings=[EmbeddingData(**data) for data in embeddings_data]
        )
        
        # Atualizar status do job
        jobs_storage[job_id]["status"] = "completed"
        jobs_storage[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_storage[job_id]["result"] = result
        
        logger.info(f"[Job {job_id}] Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Processing failed: {str(e)}", exc_info=True)
        
        # Atualizar status do documento para falhou
        repo.update_document_status(document_id, DocumentStatus.FAILED, str(e))
        
        # Limpar arquivo temporário
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        # Atualizar status do job com erro
        jobs_storage[job_id]["status"] = "failed"
        jobs_storage[job_id]["completed_at"] = datetime.now().isoformat()
        jobs_storage[job_id]["error"] = str(e)
    finally:
        db.close()
