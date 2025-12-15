"""
Cliente de API de Documentos - Trata comunicação com a API de gerenciamento de documentos.
Separa lógica de API da lógica de agente para melhor modularidade e testabilidade.
"""
import httpx
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DocumentAPIClient:
    """Cliente para interagir com a API de gerenciamento de documentos."""
    
    def __init__(self, api_base_url: str = None):
        """
        Inicializar cliente de API de Documentos.
        
        Args:
            api_base_url: URL base da API (padrão: variável de ambiente DOCUMENT_SERVICE_URL ou http://localhost:8008)
        """
        self.api_base_url = api_base_url or os.getenv("DOCUMENT_SERVICE_URL", "http://localhost:8008")
        self.timeout = 10.0
        logger.info(f"[Document API Client] Inicializado com URL: {self.api_base_url}")
    
    def fetch_documents(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """
        Buscar documentos do endpoint da API.
        """
        logger.info(f"[Document API Client] Fetching documents (status={status}, limit={limit})")
        
        try:
            params = {"limit": limit}
            if status:
                params["status"] = status
            
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(f"{self.api_base_url}/documents", params=params)
                response.raise_for_status()
                data = response.json()
            
            logger.info(f"[Document API Client] Retrieved {data.get('total', 0)} documents")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"[Document API Client] HTTP error: {str(e)}", exc_info=True)
            raise Exception(f"Failed to fetch documents from API: {str(e)}")
        except Exception as e:
            logger.error(f"[Document API Client] Error: {str(e)}", exc_info=True)
            raise


_api_client: Optional[DocumentAPIClient] = None


def get_document_api_client(api_base_url: str = None) -> DocumentAPIClient:
    """Get or create the singleton Document API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = DocumentAPIClient(api_base_url)
    return _api_client
