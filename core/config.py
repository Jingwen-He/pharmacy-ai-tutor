import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    @classmethod
    def get_api_key(cls) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_NAME: str = "claude-sonnet-4-20250514"

    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ChromaDB settings
    CHROMA_PERSIST_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed"
    )
    CHROMA_COLLECTION_NAME: str = "pharmacy_materials"

    # Chunking settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200

    # Retrieval settings
    TOP_K: int = 5

    @classmethod
    def validate(cls) -> bool:
        """Check that required settings are configured."""
        key = cls.get_api_key()
        if not key or key == "sk-ant-xxxxxxxxxxxxx":
            return False
        return True
