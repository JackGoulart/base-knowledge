"""
Agente de Listagem de Documentos - Trata consultas sobre documentos disponíveis
Usa endpoint de API em vez de acesso direto ao banco de dados
Refatorado para usar padrões modernos do LangChain com create_agent e Tools
"""
import logging
from langchain_core.tools import Tool
from langchain.agents import create_agent
from services.document_api_client import get_document_api_client
from services.openai_client import create_chat_llm
from .schema import DocumentInfo, DocumentListResponse

logger = logging.getLogger(__name__)


class DocumentListAgent:
    """Agente para tratar listagem de documentos e consultas de metadados usando agentes LangChain"""
    
    def __init__(
        self,
        api_base_url: str = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.api_client = get_document_api_client(api_base_url)
        self.llm = create_chat_llm(model_name=model_name, temperature=temperature)
        
        # Criar ferramenta que busca documentos e retorna dados estruturados
        self.tools = [
            Tool(
                name="list_documents",
                func=self._list_documents_tool,
                description=f"""List documents in the knowledge base. 
                Can filter by status (completed, processing, failed).
                Returns JSON with document details following this schema:
                {DocumentInfo.model_json_schema()}
                Use this when user asks about available documents, uploaded files, or document status.""",
            )
        ]
        
        # Criar agente usando API moderna do LangChain
        self.agent = create_agent(self.llm, self.tools)
    
    def _list_documents_tool(self, query: str) -> str:
        """Função de ferramenta que busca documentos e retorna JSON estruturado."""
        
        # Fetch documents using API client
        documents_data = self.api_client.fetch_documents()
        
        if not documents_data.get("documents"):
            return "No documents found in the knowledge base."
        
        # Retornar JSON estruturado para o LLM formatar naturalmente
        return documents_data
    
    def handle_query(self, query: str) -> DocumentListResponse:
        """
        Tratar consulta de listagem de documentos usando agente LangChain.
        
        Args:
            query: Pergunta do usuário sobre documentos
            
        Returns:
            DocumentListResponse com dados estruturados
        """
        logger.info(f"[Document List Agent] Processing query: {query}")
        
        try:
            # Deixar o agente tratar a consulta com a ferramenta list_documents
            result = self.agent.invoke({"messages": [("user", query)]})
            
            # Extrair resposta do agente
            final_response = result.get("messages", [])[-1].content if result.get("messages") else str(result)
            return DocumentListResponse(
                response=final_response,
                documents=[],
                total_documents=0
            )
            
        except Exception as e:
            logger.error(f"[Document List Agent] Error processing query: {str(e)}", exc_info=True)
            return DocumentListResponse(
                response=f"I apologize, but I encountered an error while retrieving the document list: {str(e)}",
                documents=[],
                total_documents=0,
                error=str(e)
            )


_document_list_agent = None

def get_document_list_agent() -> DocumentListAgent:
    """instância singleton do agente de lista de documentos"""
    global _document_list_agent
    if _document_list_agent is None:
        _document_list_agent = DocumentListAgent()
    return _document_list_agent
