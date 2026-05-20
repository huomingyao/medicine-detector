"""
Milvus helper module for medicine bottle vector storage and retrieval.
Tries Milvus first, falls back to simple JSON store.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Milvus, fall back to simple store
try:
    from pymilvus import MilvusClient, DataType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    from simple_store import SimpleVectorStore as FallbackStore
else:
    from simple_store import SimpleVectorStore as FallbackStore
    from config import (
        MILVUS_URI,
        COLLECTION_NAME,
        CLIP_VECTOR_DIM,
        INDEX_METRIC_TYPE,
        INDEX_M,
        INDEX_EF_CONSTRUCTION,
    )


class MilvusHelper:
    """
    Milvus database helper. Uses real Milvus if available,
    falls back to simple JSON-based store otherwise.
    """

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        """Connect to Milvus or fall back to simple store."""
        if MILVUS_AVAILABLE:
            try:
                logger.info(f"Connecting to Milvus: {MILVUS_URI}")
                self._client = MilvusClient(uri=MILVUS_URI)
                logger.info("Connected to Milvus successfully")

                if not self._client.has_collection(COLLECTION_NAME):
                    logger.info(f"Creating collection: {COLLECTION_NAME}")
                    self._create_collection()

            except Exception as e:
                logger.warning(f"Milvus not available: {e}, using simple store")
                self._use_fallback()
        else:
            logger.info("Using simple vector store (fallback)")
            self._use_fallback()

    def _use_fallback(self):
        """Use simple JSON-based store."""
        self._client = FallbackStore()

    def _create_collection(self):
        """Create the collection with HNSW index."""
        schema = self._client.create_schema(auto_id=True, vector_field_name="vector")

        schema.add_field(name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(name="image_path", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(name="text_label", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(name="vector", datatype=DataType.FLOAT_VECTOR, dim=CLIP_VECTOR_DIM)

        index_params = {
            "metric_type": INDEX_METRIC_TYPE,
            "index_type": "HNSW",
            "params": {
                "M": INDEX_M,
                "efConstruction": INDEX_EF_CONSTRUCTION,
            },
        }

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

    def insert_sample(self, image_path: str, text_label: str, vector: np.ndarray) -> int:
        """Insert a sample."""
        return self._client.insert_sample(image_path, text_label, vector)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar samples."""
        return self._client.search(query_vector, top_k)

    def get_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """Get a sample by ID."""
        return self._client.get_sample(sample_id)

    def delete_sample(self, sample_id: int) -> bool:
        """Delete a sample."""
        return self._client.delete_sample(sample_id)

    def list_samples(self, keyword: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List samples."""
        return self._client.list_samples(keyword, limit)

    def get_collection_name(self) -> str:
        return self._client.get_collection_name()

    def get_total_count(self) -> int:
        return self._client.get_total_count()


def create_collection():
    """Create or reset collection."""
    helper = MilvusHelper()
    if hasattr(helper._client, '_data'):
        helper._client._data = {"samples": [], "next_id": 1}
        helper._client._vectors = np.array([])
        helper._client._save()
    return helper


def delete_collection():
    """Delete collection."""
    from config import BASE_DIR
    data_dir = BASE_DIR / "data"
    if (data_dir / f"{COLLECTION_NAME}.json").exists():
        (data_dir / f"{COLLECTION_NAME}.json").unlink()
    if (data_dir / f"{COLLECTION_NAME}_vectors.npy").exists():
        (data_dir / f"{COLLECTION_NAME}_vectors.npy").unlink()


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python milvus_helper.py [create|delete|stats]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        create_collection()
    elif command == "delete":
        delete_collection()
    elif command == "stats":
        helper = MilvusHelper()
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total: {helper.get_total_count()}")