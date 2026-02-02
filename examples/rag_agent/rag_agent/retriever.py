"""Retriever abstraction for vector stores."""

from pathlib import Path
from typing import Any, Protocol

from sentence_transformers import SentenceTransformer

from rag_agent.models import Document


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
    """ChromaDB-based retriever with sentence transformer embeddings."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str | Path = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedder = SentenceTransformer(embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the sentence transformer."""
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

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
                # Convert distance to similarity score (cosine distance -> similarity)
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
            # Generate IDs based on current count
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
        # Delete and recreate the collection
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
