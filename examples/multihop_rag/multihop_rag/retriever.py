"""Retriever abstraction for vector stores, adapted for news article metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from multihop_rag.models import Document

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class Retriever(Protocol):
    """Protocol for retriever implementations."""

    def query(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve documents matching the query."""
        ...

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the store."""
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


class ChromaRetriever:
    """ChromaDB-based retriever using Fireworks AI embeddings.

    Adapted for news article metadata schema:
    title, author, category, published_at, source, url.

    Requires FIREWORKS_API_KEY environment variable.
    """

    INIT_MARKER = ".initialized"

    def __init__(
        self,
        collection_name: str = "multihop_rag",
        persist_directory: str | Path = "./chroma_db",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        import chromadb
        from chromadb.config import Settings
        from openai import OpenAI

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            msg = (
                "FIREWORKS_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )
            raise RuntimeError(msg)

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._marker_path = self.persist_directory / self.INIT_MARKER
        self._embedding_model = embedding_model

        self._openai = OpenAI(
            api_key=api_key,
            base_url=FIREWORKS_BASE_URL,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def is_initialized(self) -> bool:
        """Check if the retriever has been initialized before."""
        return self._marker_path.exists()

    def mark_initialized(self) -> None:
        """Mark the retriever as initialized."""
        self._marker_path.touch()

    def clear_initialization(self) -> None:
        """Clear the initialization marker."""
        if self._marker_path.exists():
            self._marker_path.unlink()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the Fireworks AI embeddings API."""
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def query(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve documents matching the query."""
        query_embedding = self._embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["ids"] and results["ids"][0]:
            for doc_id, content, distance, metadata in zip(
                results["ids"][0],
                results["documents"][0],  # type: ignore
                results["distances"][0],  # type: ignore
                results["metadatas"][0],  # type: ignore
            ):
                # Convert cosine distance to similarity score
                score = 1 - distance
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        score=score,
                        metadata=metadata or {},
                    )
                )

        return documents

    def query_by_category(
        self,
        query: str,
        category: str,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve documents filtered by news category."""
        return self.query(query, k=k, filters={"category": category})

    def query_by_source(
        self,
        query: str,
        source: str,
        k: int = 10,
    ) -> list[Document]:
        """Retrieve documents filtered by news source name."""
        return self.query(query, k=k, filters={"source": source})

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the collection."""
        if not documents:
            return

        if ids is None:
            start_id = self.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self._embed(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
