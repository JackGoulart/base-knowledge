# Base Knowledge 

Sistema de processamento de documentos com agentes de IA, RAG (Retrieval-Augmented Generation) e busca web integrada.

## 🚀 Características

- 📄 **Processamento de Documentos**: Upload de PDF/DOCX, chunking híbrido com Docling, embeddings OpenAI
- 🤖 **Agentes Inteligentes**: Orquestração multi-agente com LangGraph para RAG, listagem de documentos e busca web
- 💬 **Interface de Chat**: Gradio web UI para interação natural
- 🗄️ **Busca Vetorial**: PostgreSQL + pgvector para busca semântica
- 🔄 **Fallback Automático**: Integração automática com busca web quando necessário

## 🏗️ Arquitetura

```
PostgreSQL + pgvector (5432)
         │
    ┌────┴────┐
    │         │
Document   Chat
Service   Service
(8008)    (8009)
          + Gradio UI
```

**Document Service**: Processa documentos, gera embeddings, armazena no PostgreSQL  
**Chat Service**: Interface de chat, orquestração de agentes, RAG e busca web

## ⚡ Quick Start

### Com Docker (Recomendado)

```bash
# 1. Configurar variáveis de ambiente
# Para OpenAI padrão:
cp env.example .env
# Edite .env e adicione sua OPENAI_API_KEY

# Para Azure OpenAI:
cp env.azure.example .env
# Edite .env e configure as variáveis do Azure

# 2. Iniciar todos os serviços
docker-compose up --build

# 3. Acessar interface Gradio
open http://localhost:8009/chat
```

### Desenvolvimento Local

```bash
# 1. Instalar dependências
uv sync

# 2. Configurar .env
# Para OpenAI padrão:
cp env.example .env

# Para Azure OpenAI:
cp env.azure.example .env

# Edite o arquivo .env com suas credenciais

# 3. Iniciar PostgreSQL
docker-compose up postgres -d

# 4. Inicializar banco
python scripts/init_db.py

# 5. Executar serviços (em terminais separados)
cd document-service && uvicorn main:app --reload --port 8008
cd chat-service && uvicorn main:app --reload --port 8009
```

## 📚 Uso

### 1. Upload de Documento
```bash
curl -X POST http://localhost:8008/documents/upload \
  -F "file=@documento.pdf" \
  -F "chunk_size=512"
```

### 2. Chat via API
```bash
curl -X POST http://localhost:8009/api/qa \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Resuma o documento",
    "session_id": "sessao-123"
  }'
```

### 3. Interface Web
Acesse `http://localhost:8009/chat` para usar a interface Gradio

## 🤖 Agentes

O sistema possui 3 agentes especializados com roteamento automático via LLM:

| Agente | Função | Exemplo de Query |
|--------|--------|------------------|
| **Document List** | Lista metadados dos documentos | "Quais documentos você tem?" |
| **RAG** | Responde usando conteúdo dos documentos | "O que o documento diz sobre X?" |
| **Web Search** | Busca informações na web via DuckDuckGo | "Últimas notícias sobre IA" |

O **fallback automático** aciona busca web quando o RAG não encontra resultados suficientes.

## 🛠️ Stack Tecnológico

- **FastAPI** + **Uvicorn** - Backend
- **Docling** - Processamento de documentos
- **PostgreSQL** + **pgvector** - Banco de dados vetorial
- **OpenAI** - Embeddings e LLM
- **LangGraph** + **LangChain** - Orquestração de agentes
- **Gradio** - Interface web
- **Docker** - Containerização

## 📁 Estrutura

```
base-knowledge/
├── shared/              # Modelos e repositórios compartilhados
├── document-service/    # Serviço de processamento de documentos
├── chat-service/        # Serviço de chat e agentes
├── scripts/             # Scripts utilitários
├── docker-compose.yml
└── pyproject.toml
```

### 4. APIs Doc
Acesse `http://localhost:8009/docs` para API chat 
Acesse `http://localhost:8008/docs` para API Documentos 



OBS: O Projeto foi testado via API OpenAI não foi testado com AzureOpenAI services

