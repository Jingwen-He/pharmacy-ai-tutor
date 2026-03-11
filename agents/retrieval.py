"""Retrieval Agent: searches the vector database for relevant teaching material chunks."""

from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore


class RetrievalAgent:
    """Searches ChromaDB for the most relevant chunks from teaching material PDFs."""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()

    def ingest_pdf(self, pdf_path: str) -> int:
        """Process a PDF and store its chunks in the vector database.

        Returns the number of chunks stored.
        """
        chunks = self.pdf_processor.process_pdf(pdf_path)
        count = self.vector_store.add_chunks(chunks)
        return count

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for relevant chunks given a query.

        Returns a list of dicts with keys:
            text, page_number, section_title, source, chunk_id, relevance_score
        """
        return self.vector_store.search(query, top_k=top_k)

    def get_chunk_count(self) -> int:
        """Return the number of chunks in the vector store."""
        return self.vector_store.get_chunk_count()

    def clear(self):
        """Clear all stored chunks."""
        self.vector_store.clear_collection()

    def get_sections(self) -> list[str]:
        """Return a list of unique section titles from stored chunks."""
        try:
            all_data = self.vector_store.collection.get(include=["metadatas"])
            if all_data["metadatas"]:
                sections = sorted(
                    set(m["section_title"] for m in all_data["metadatas"])
                )
                return sections
        except Exception:
            pass
        return []
