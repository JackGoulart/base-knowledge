import sys
from pathlib import Path 
sys.path.insert(0, str(Path.cwd().parent))

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import gradio as gr
from dotenv import load_dotenv

from shared.database.config import init_db
from routes import chat, conversations

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context manager para eventos de inicialização/encerramento
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializar banco de dados na inicialização da aplicação"""
    logger.info("Inicializando banco de dados...")
    init_db()
    logger.info("Banco de dados inicializado com sucesso")
    yield

# Inicializar aplicação FastAPI com lifespan
app = FastAPI(
    title="Serviço de Chat",
    description="API para Q&A baseado em documentos usando agentes de IA e RAG",
    lifespan=lifespan
)

# Incluir routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])

# Montar interface Gradio
chat_interface = chat.create_gradio_interface()
app = gr.mount_gradio_app(app, chat_interface, path="/chat")


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    return {
        "status": "healthy",
        "service": "chat-service",
        "port": 8009
    }
