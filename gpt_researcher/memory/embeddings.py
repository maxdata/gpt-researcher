from langchain_community.vectorstores import FAISS
import os

OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")


class Memory:
    def __init__(self, embedding_provider, headers=None, **kwargs):
        _embeddings = None
        headers = headers or {}
        match embedding_provider:            
            case "openai":
                from langchain_openai import OpenAIEmbeddings 
                _embeddings = OpenAIEmbeddings(
                    openai_api_key=headers.get("openai_api_key")
                    or os.environ.get("OPENAI_API_KEY"),
                    model=OPENAI_EMBEDDING_MODEL
                )
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    deployment='text-embedding-3-small', chunk_size=16
                )
            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
