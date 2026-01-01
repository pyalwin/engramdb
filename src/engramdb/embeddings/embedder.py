"""
Embedder: Generate vector embeddings for text content.

Supports multiple embedding backends for flexibility between
proprietary (OpenAI) and open-source (sentence-transformers) models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import os


class Embedder(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIEmbedder(Embedder):
    """
    OpenAI embedding model.

    Supported models:
    - text-embedding-3-small (1536 dims, cheapest)
    - text-embedding-3-large (3072 dims, best quality)
    - text-embedding-ada-002 (1536 dims, legacy)
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: uv add openai")

        self.model = model
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        OpenAI supports batching natively, which is more efficient.
        """
        if not texts:
            return []

        # Filter out empty texts, truncate long texts, and track original indices
        # OpenAI text-embedding-3-small has 8191 token limit (~32k chars approx)
        MAX_CHARS = 30000  # Conservative limit to stay under token limit
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            cleaned = text.strip() if text else ""
            if cleaned:
                # Truncate if too long
                if len(cleaned) > MAX_CHARS:
                    cleaned = cleaned[:MAX_CHARS]
                valid_texts.append(cleaned)
                valid_indices.append(i)

        if not valid_texts:
            # Return zero vectors for all empty inputs
            return [[0.0] * self._dimension for _ in texts]

        # OpenAI has a limit of ~8000 tokens per request
        # Split into chunks if needed
        max_batch_size = 100
        valid_embeddings = []

        for i in range(0, len(valid_texts), max_batch_size):
            batch = valid_texts[i:i + max_batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch
            )
            # Results are returned in the same order as input
            batch_embeddings = [item.embedding for item in response.data]
            valid_embeddings.extend(batch_embeddings)

        # Reconstruct full list with zero vectors for empty inputs
        all_embeddings = [[0.0] * self._dimension for _ in texts]
        for idx, embedding in zip(valid_indices, valid_embeddings):
            all_embeddings[idx] = embedding

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbedder(Embedder):
    """
    Mock embedder for testing without API calls.

    Generates deterministic pseudo-embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize mock embedder.

        Args:
            dimension: Embedding dimension to generate
        """
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text."""
        import hashlib

        # Create deterministic hash from text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Generate embedding from hash
        embedding = []
        for i in range(self._dimension):
            # Use different parts of the hash to generate values
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] + i) / 255.0  # Normalize to 0-1
            value = (value - 0.5) * 2  # Normalize to -1 to 1
            embedding.append(value)

        # Normalize to unit vector
        import math
        magnitude = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


# Optional: Local embedder using sentence-transformers
class LocalEmbedder(Embedder):
    """
    Local embedding model using sentence-transformers.

    Recommended models:
    - intfloat/e5-base-v2 (768 dims, good quality)
    - all-MiniLM-L6-v2 (384 dims, fast)
    - BAAI/bge-small-en-v1.5 (384 dims, good for retrieval)
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        """
        Initialize local embedder.

        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: uv add sentence-transformers"
            )

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedder(
    backend: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> Embedder:
    """
    Factory function to create an embedder.

    Args:
        backend: "openai", "local", or "mock"
        model: Model name (optional, uses defaults)
        **kwargs: Additional arguments passed to embedder

    Returns:
        Embedder instance
    """
    if backend == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIEmbedder(model=model, **kwargs)
    elif backend == "local":
        model = model or "intfloat/e5-base-v2"
        return LocalEmbedder(model_name=model)
    elif backend == "mock":
        return MockEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'openai', 'local', or 'mock'.")
