
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

# Store the local vector database inside the project's data folder.
CHROMA_DIRECTORY = str(
    Path(settings.document_registry_db_path).parent / "chroma_db"
)
COLLECTION_NAME = "enterprise_it_support"


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )

    return _embeddings


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        Path(CHROMA_DIRECTORY).mkdir(parents=True, exist_ok=True)

        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIRECTORY,
        )

    return _vectorstore


def get_retriever():
    from app.services.document_registry import get_document_registry

    active_version_ids = (
        get_document_registry().get_active_version_ids()
    )

    search_kwargs = {"k": settings.top_k}

    if active_version_ids:
        search_kwargs["filter"] = {
            "version_id": {"$in": active_version_ids}
        }

    return get_vectorstore().as_retriever(
        search_kwargs=search_kwargs
    )


def add_documents(chunks, ids=None):
    store = get_vectorstore()

    # Chroma does not accept None or complex objects as metadata values.
    clean_chunks = []

    for chunk in chunks:
        metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if value is not None
            and isinstance(value, (str, int, float, bool))
        }

        clean_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    if ids is None:
        return store.add_documents(clean_chunks)

    return store.add_documents(clean_chunks, ids=ids)
