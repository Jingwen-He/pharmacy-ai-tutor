"""ChromaDB vector store operations for embedding storage and semantic search."""

import chromadb
from sentence_transformers import SentenceTransformer

from .config import Settings


class VectorStore:
    """Manages ChromaDB for storing and searching document embeddings."""

    def __init__(self):
        self.embedding_model = SentenceTransformer(Settings.EMBEDDING_MODEL)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=Settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of dicts with keys: text, page_number, section_title, source, chunk_id

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [
            {
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "source": chunk["source"],
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()

        # Add to ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for the most relevant chunks given a query.

        Args:
            query: The search query string.
            top_k: Number of results to return (defaults to Settings.TOP_K).

        Returns:
            List of dicts with keys: text, page_number, section_title, source,
            chunk_id, relevance_score.
        """
        if top_k is None:
            top_k = Settings.TOP_K

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # ChromaDB returns distances; convert to similarity score
                distance = results["distances"][0][i]
                relevance_score = 1 - distance  # cosine distance to similarity

                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                search_results.append(
                    {
                        "text": results["documents"][0][i],
                        "page_number": meta.get("page_number", 0),
                        "section_title": meta.get("section_title", "Unknown"),
                        "source": meta.get("source", "Unknown"),
                        "chunk_id": results["ids"][0][i],
                        "relevance_score": round(relevance_score, 4),
                    }
                )

        return search_results

    def clear_collection(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(Settings.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=Settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_chunk_count(self) -> int:
        """Return the number of chunks currently stored."""
        return self.collection.count()
