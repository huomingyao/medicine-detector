"""
Simple vector store using local JSON files.
Fallback for when Milvus is not available.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from config import (
    BASE_DIR,
    COLLECTION_NAME,
    CLIP_VECTOR_DIM,
)

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class SimpleVectorStore:
    """
    Simple vector store using JSON and NumPy.
    Stores vectors locally, suitable for small-scale applications.
    """

    _instance = None
    _data_file = None
    _vectors_file = None
    _data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._data is None:
            self._data_file = DATA_DIR / f"{COLLECTION_NAME}.json"
            self._vectors_file = DATA_DIR / f"{COLLECTION_NAME}_vectors.npy"
            self._load()

    def _load(self):
        """Load data from files."""
        # Load metadata
        if self._data_file.exists():
            with open(self._data_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"samples": [], "next_id": 1}

        # Load vectors
        if self._vectors_file.exists():
            self._vectors = np.load(str(self._vectors_file))
        else:
            self._vectors = np.array([])

        logger.info(f"Loaded {len(self._data['samples'])} samples")

    def _save(self):
        """Save data to files."""
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

        if len(self._vectors) > 0:
            np.save(str(self._vectors_file), self._vectors)

    def insert_sample(
        self, image_path: str, text_label: str, vector: np.ndarray
    ) -> int:
        """Insert a sample."""
        sample_id = self._data["next_id"]

        sample = {
            "id": sample_id,
            "image_path": image_path,
            "text_label": text_label,
            "vector_id": len(self._vectors),
        }

        self._data["samples"].append(sample)
        self._data["next_id"] += 1

        # Append vector
        if len(self._vectors) == 0:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector])

        self._save()

        logger.info(f"Inserted sample with ID: {sample_id}")
        return sample_id

    def search(
        self, query_vector: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for similar samples using cosine similarity."""
        if len(self._vectors) == 0:
            return []

        # Normalize vectors
        query_norm = query_vector / np.linalg.norm(query_vector)
        vectors_norm = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(vectors_norm, query_norm)

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self._data["samples"]):
                sample = self._data["samples"][idx]
                results.append({
                    "id": sample["id"],
                    "image_path": sample["image_path"],
                    "text_label": sample["text_label"],
                    "score": float(similarities[idx]),
                })

        return results

    def get_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """Get a sample by ID."""
        for sample in self._data["samples"]:
            if sample["id"] == sample_id:
                return sample
        return None

    def delete_sample(self, sample_id: int) -> bool:
        """Delete a sample (mark as deleted)."""
        for sample in self._data["samples"]:
            if sample["id"] == sample_id:
                sample["deleted"] = True
                self._save()
                return True
        return False

    def list_samples(self, keyword: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List samples."""
        results = []
        for sample in self._data["samples"]:
            if sample.get("deleted"):
                continue
            if keyword:
                if keyword.lower() not in sample["text_label"].lower():
                    continue
            results.append({
                "id": sample["id"],
                "image_path": sample["image_path"],
                "text_label": sample["text_label"],
            })
            if len(results) >= limit:
                break

        return results

    def get_collection_name(self) -> str:
        return COLLECTION_NAME

    def get_total_count(self) -> int:
        return sum(1 for s in self._data["samples"] if not s.get("deleted", False))


# Alias for compatibility
MilvusHelper = SimpleVectorStore


def create_collection():
    """Reset/clear the collection."""
    helper = SimpleVectorStore()
    helper._data = {"samples": [], "next_id": 1}
    helper._vectors = np.array([])
    helper._save()
    logger.info("Collection created/reset successfully")
    return helper


def delete_collection():
    """Delete all data."""
    if SimpleVectorStore._data_file.exists():
        SimpleVectorStore._data_file.unlink()
    if SimpleVectorStore._vectors_file.exists():
        SimpleVectorStore._vectors_file.unlink()
    logger.info("Collection deleted")


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python simple_store.py [create|delete|stats]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        create_collection()
        print("Collection created successfully")
    elif command == "delete":
        delete_collection()
        print("Collection deleted successfully")
    elif command == "stats":
        helper = SimpleVectorStore()
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total entities: {helper.get_total_count()}")
    else:
        print(f"Unknown command: {command}")