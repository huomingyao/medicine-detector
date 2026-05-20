"""
Chinese-CLIP vectorizer module.
Converts images to 768-dimensional vectors using the chinese-clip-vit-large-patch14 model.
"""

import numpy as np
from PIL import Image
from typing import Union
from transformers import CLIPModel, CLIPImageProcessor
import torch

from config import CLIP_MODEL_NAME, CLIP_VECTOR_DIM


class CLIPVectorizer:
    """Chinese-CLIP image vectorizer."""

    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            print(f"Loading Chinese-CLIP model: {CLIP_MODEL_NAME}")
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self._processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
            self._model.eval()
            print("Model loaded successfully")

    def image_to_vector(self, image_path_or_pil: Union[str, Image.Image]) -> np.ndarray:
        """
        Convert an image to a 768-dimensional normalized vector.

        Args:
            image_path_or_pil: Image file path or PIL Image object

        Returns:
            Normalized 768-dimensional numpy array
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        # Preprocess with CLIPImageProcessor
        inputs = self._processor(images=image, return_tensors="pt")

        with torch.no_grad():
            # Get image features
            image_features = self._model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()


def image_to_vector(image_path_or_pil: Union[str, Image.Image]) -> np.ndarray:
    """
    Convenience function to convert an image to a vector.

    Args:
        image_path_or_pil: Image file path or PIL Image object

    Returns:
        Normalized 768-dimensional numpy array
    """
    vectorizer = CLIPVectorizer()
    return vectorizer.image_to_vector(image_path_or_pil)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vectorizer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")

    vector = image_to_vector(image_path)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector sample (first 10 values): {vector[:10]}")
    print(f"Vector norm: {np.linalg.norm(vector)}")
    print("Vectorization successful!")