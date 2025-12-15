import sys
from pathlib import Path 
sys.path.insert(0, str(Path.cwd().parent))

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from shared.database.config import init_db
from routes import upload, documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Armazenamento de jobs (para coordenação de tarefas em background)
jobs_storage: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializar banco de dados na inicialização da aplicação"""
    logger.info("Inicializando banco de dados...")
    init_db()
    logger.info("Banco de dados inicializado com sucesso")
    app.state.jobs_storage = jobs_storage
    logger.info("Jobs storage inicializado")
    
    yield

# Inicializar aplicação FastAPI com lifespan
app = FastAPI(
    title="Serviço de Documentos",
    description="Upload de arquivos PDF/DOCX para conversão, chunking e geração de embeddings",
    lifespan=lifespan
)

# Incluir routers
app.include_router(upload.router, prefix="/documents", tags=["upload"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
