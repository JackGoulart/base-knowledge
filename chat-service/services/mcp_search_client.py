import logging
import asyncio
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPSearchClient:
    """Cliente para realizar buscas web via servidor MCP DuckDuckGo local."""
    
    def __init__(self):
        """
        Inicializa o cliente de busca MCP.
        
        O servidor MCP DuckDuckGo é executado localmente via stdio.
        """
        # Configurar parâmetros para executar o servidor MCP localmente
        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "duckduckgo_mcp_server.server"],
            env=None
        )
    
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Realiza busca web síncrona.
        """
        try:
            return asyncio.run(self._search_async(query, max_results))
        except Exception as e:
            logger.error(f"[MCP Search] Erro na busca: {e}")
            return f"Erro na busca: {str(e)}"
    
    async def search_async(self, query: str, max_results: int = 5) -> str:
        """
        Realiza busca web assíncrona.
        """
        return await self._search_async(query, max_results)
    
    async def _search_async(self, query: str, max_results: int) -> str:
        """Implementação interna de busca assíncrona."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "search",
                        arguments={"query": query, "max_results": max_results}
                    )
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    return "Nenhum resultado encontrado"
        except Exception as e:
            logger.error(f"[MCP Search] Erro na busca assíncrona: {e}")
            raise


_mcp_client: Optional[MCPSearchClient] = None


def get_mcp_search_client() -> MCPSearchClient:
    """Get or create the singleton MCP search client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPSearchClient()
    return _mcp_client
