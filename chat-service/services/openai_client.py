import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings


def _is_azure() -> bool:
    """Verifica se Azure OpenAI deve ser usado com base nas variáveis de ambiente."""
    return os.getenv("AZURE_OPENAI_ENDPOINT") is not None


def create_chat_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: str = None,
    base_url: str = None
) -> ChatOpenAI:
    """
    Cria uma instância ChatOpenAI ou AzureChatOpenAI baseado na configuração.
    """
    if _is_azure():
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", model_name),
            temperature=temperature
        )
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )


def create_embeddings(
    model_name: str = "text-embedding-3-small",
    api_key: str = None,
    base_url: str = None
) -> OpenAIEmbeddings:
    """
    Cria uma instância OpenAIEmbeddings ou AzureOpenAIEmbeddings baseado na configuração.
    """
    if _is_azure():
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    else:
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
