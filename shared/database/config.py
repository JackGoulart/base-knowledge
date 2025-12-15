"""
Database configuration and session management
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/base_knowledge"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False 
)

# Criar classe SessionLocal
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Criar classe Base para models
Base = declarative_base()


def get_db():
    """
    Função de dependência para obter sessão do banco de dados.
    Use com FastAPI Depends.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Inicializar tabelas do banco de dados.
    Deve ser chamado na inicialização da aplicação.
    """
    # Importar todos os models aqui para garantir que sejam registrados com Base
    from shared.models import Document, DocumentChunk
    from shared.models.conversation import Conversation, Message
    import logging
    logger = logging.getLogger(__name__)
    
    # Criar extensão pgvector se não existir
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Criar todas as tabelas (índices normais serão criados automaticamente)
    Base.metadata.create_all(bind=engine)
    
    # Criar índice vetorial HNSW
    try:
        with engine.connect() as conn:
            # Verificar se já existe
            result = conn.execute(text("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'idx_document_chunks_embedding_hnsw'
            """))
            
            if not result.fetchone():
                logger.info("Criando índice vetorial HNSW para embeddings...")
                
                conn.execute(text("""
                    CREATE INDEX idx_document_chunks_embedding_hnsw 
                    ON document_chunks 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64)
                """))
                conn.commit()
                logger.info("Índice vetorial HNSW criado com sucesso!")
            else:
                logger.info("Índice vetorial HNSW já existe")
    except Exception as e:
        # Pode falhar se pgvector não suportar HNSW (< 0.5.0)
        logger.warning(f"Não foi possível criar índice HNSW: {e}")
        logger.info("Execute scripts/create_vector_index.sql manualmente para melhor performance")
