import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Create directories
CROPS_DIR = BASE_DIR / "crops"
UPLOADS_DIR = BASE_DIR / "uploads"
CROPS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Milvus configuration (use Milvus Lite for local testing)
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_lite.db")
COLLECTION_NAME = "medicine_bottles"

# YOLO model configuration
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "d:/api_shibie/yolov8x-v8.2.0.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

# Chinese-CLIP configuration - use local model
CLIP_MODEL_NAME = "./models/chinese-clip-vit-large-patch14"
CLIP_VECTOR_DIM = 768

# HNSW index parameters
INDEX_METRIC_TYPE = "COSINE"
INDEX_M = 16
INDEX_EF_CONSTRUCTION = 200

# DashScope API configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen-vl-plus-latest"

# Recognition parameters
SIMILARITY_THRESHOLD = 0.7

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Image parameters
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}