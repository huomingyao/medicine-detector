"""
FastAPI main application for medicine bottle recognition.
"""

import os
import io
import json
import uuid
import time
import logging
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import (
    CROPS_DIR,
    UPLOADS_DIR,
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_SIZE,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
)
from models import (
    SampleCreate,
    SampleResponse,
    SampleListResponse,
    RecognizeResponse,
    MedicineInfo,
    MatchedSample,
    ErrorResponse,
)
from vectorizer import image_to_vector as clip_image_to_vector
from milvus_helper import MilvusHelper
from llm_service import LLMService
from yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="药瓶识别系统",
    description="基于向量检索与多模态大模型的药瓶识别 API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    request.state.start_time = time.time()

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    elapsed = time.time() - request.state.start_time
    logger.info(
        f"[{request_id}] Completed in {elapsed:.2f}s - {response.status_code}"
    )

    return response


# Initialize services
milvus_helper: Optional[MilvusHelper] = None
llm_service: Optional[LLMService] = None
yolo_detector: Optional[YOLODetector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global milvus_helper, llm_service, yolo_detector

    logger.info("Initializing services...")

    # Initialize Milvus
    try:
        milvus_helper = MilvusHelper()
        logger.info("Milvus initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        raise

    # Initialize LLM service
    try:
        llm_service = LLMService()
        logger.info("LLM service initialized successfully")
    except ValueError as e:
        # LLM service may not be ready, but don't fail
        logger.warning(f"LLM service not initialized: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM service: {e}")

    # Initialize YOLO detector
    try:
        if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
            yolo_detector = YOLODetector(YOLO_MODEL_PATH, YOLO_CONFIDENCE)
            logger.info(f"YOLO detector initialized: {YOLO_MODEL_PATH}")
        else:
            logger.warning(f"YOLO model not found: {YOLO_MODEL_PATH}, skipping YOLO")
    except Exception as e:
        logger.warning(f"Failed to initialize YOLO detector: {e}")

    logger.info("Application started successfully")


def validate_image(file: UploadFile) -> Image.Image:
    """
    Validate and load an image file.

    Args:
        file: UploadFile object

    Returns:
        PIL Image object
    """
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Load image
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))

        # Check image size
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {MAX_IMAGE_SIZE / 1024 / 1024}MB",
            )

        # Reset file pointer
        file.file.seek(0)

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


def save_image(image: Image.Image, prefix: str = "img", subdir: Path = None) -> str:
    """
    Save image to specified directory.

    Args:
        image: PIL Image object
        prefix: Filename prefix
        subdir: Subdirectory (uploads/crops)

    Returns:
        Saved file path
    """
    target_dir = subdir if subdir else UPLOADS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:4]
    filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
    filepath = target_dir / filename

    # Save
    image.save(filepath, "JPEG", quality=95)

    return str(filepath)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "药瓶识别系统",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    health_status = {"status": "healthy"}

    # Check Milvus
    try:
        if milvus_helper:
            count = milvus_helper.get_total_count()
            health_status["milvus"] = {"status": "healthy", "samples": count}
        else:
            health_status["milvus"] = {"status": "not_initialized"}
    except Exception as e:
        health_status["milvus"] = {"status": "error", "message": str(e)}

    # Check LLM
    health_status["llm"] = {"status": "available" if llm_service else "not_configured"}

    return health_status


# ==================== Sample Management API ====================


@app.post(
    "/api/v1/samples",
    response_model=SampleResponse,
    summary="添加样本",
    description="向向量数据库中添加药瓶样本图片和标签",
)
async def add_sample(
    file: UploadFile = File(..., description="药瓶图片文件"),
    text_label: str = Form(..., description="药品标签，如 阿莫西林胶囊 0.5g 白云山制药"),
    use_yolo: bool = Form(True, description="是否使用YOLO裁剪药瓶区域"),
):
    """
    Add a new sample to the database.

    - **file**: Image file (jpg, png, webp)
    - **text_label**: Medicine label text
    - **use_yolo**: 是否使用YOLO裁剪（默认开启）
    """
    request_id = getattr(app.extra.get("state", {}), "request_id", "unknown")

    try:
        # Validate and load image
        logger.info(f"[{request_id}] Loading image...")
        image = validate_image(file)

        # Generate IDs for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:4]

        # Save original to uploads directory
        original_image_path = save_image(image, "sample", UPLOADS_DIR)

        # YOLO crop if enabled and detector available
        image_path = original_image_path
        if use_yolo and yolo_detector:
            logger.info(f"[{request_id}] Running YOLO detection...")
            yolo_start = time.time()
            detections = yolo_detector.detect(original_image_path)
            yolo_elapsed = time.time() - yolo_start
            logger.info(f"[{request_id}] YOLO detected {len(detections)} bottles in {yolo_elapsed:.2f}s")

            if detections and len(detections) > 0:
                # Use the first (highest confidence) detection
                det = detections[0]
                logger.info(f"[{request_id}] Cropping bottle region, YOLO confidence: {det['confidence']:.2f}")

                # Crop the bottle region
                crop_img = yolo_detector.crop_bottle(original_image_path, det['bbox'], margin=10)
                from PIL import Image as PILImage
                crop_pil = PILImage.fromarray(crop_img)

                # Save cropped image to crops directory
                crop_filename = f"crop_{timestamp}_{unique_id}.jpg"
                crop_path = CROPS_DIR / crop_filename
                crop_pil.save(crop_path)
                image_path = str(crop_path)
                logger.info(f"[{request_id}] Cropped and saved: {image_path}")
        elif use_yolo and not yolo_detector:
            logger.warning(f"[{request_id}] YOLO not available, saving original image")

        # Generate vector
        logger.info(f"[{request_id}] Generating vector...")
        vector = clip_image_to_vector(image_path if image_path != original_image_path else image)

        # Insert to Milvus
        logger.info(f"[{request_id}] Inserting to Milvus...")
        sample_id = milvus_helper.insert_sample(image_path, text_label, vector)

        logger.info(f"[{request_id}] Sample added successfully: {sample_id}")

        return SampleResponse(
            id=sample_id,
            image_path=image_path,
            text_label=text_label,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error adding sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/samples",
    response_model=SampleListResponse,
    summary="获取样本列表",
    description="获取已入库的药瓶样本列表",
)
async def list_samples(
    keyword: Optional[str] = Query(None, description="关键词过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
):
    """
    List samples with optional keyword filter.

    - **keyword**: Optional keyword to filter labels
    - **limit**: Maximum number of results (default 100)
    """
    try:
        samples = milvus_helper.list_samples(keyword=keyword, limit=limit)

        return SampleListResponse(
            samples=[
                SampleResponse(
                    id=s["id"],
                    image_path=s["image_path"],
                    text_label=s["text_label"],
                )
                for s in samples
            ],
            total=len(samples),
        )

    except Exception as e:
        logger.error(f"Error listing samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/v1/samples/{sample_id}",
    response_model=ErrorResponse,
    summary="删除样本",
    description="从向量数据库中删除指定样本",
)
async def delete_sample(sample_id: int):
    """
    Delete a sample by ID.

    Note: Milvus Lite has limited support for deletion.
    """
    try:
        success = milvus_helper.delete_sample(sample_id)

        if success:
            return {"success": True, "message": f"Sample {sample_id} deleted"}
        else:
            return {
                "success": False,
                "message": f"Delete not fully supported in Milvus Lite",
            }

    except Exception as e:
        logger.error(f"Error deleting sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Recognition API ====================


@app.post(
    "/api/v1/recognize",
    response_model=RecognizeResponse,
    summary="识别药瓶",
    description="使用YOLO检测药瓶并用Qwen-VL识别(支持多药瓶)",
)
async def recognize(
    file: UploadFile = File(..., description="待识别的药瓶图片"),
):
    """
    Recognize medicine bottles using YOLO + Qwen-VL.
    """
    request_id = getattr(app.extra.get("state", {}), "request_id", "unknown")

    try:
        start_time = time.time()

        # Step 1: Validate and load image
        logger.info(f"[{request_id}] Loading recognition image...")
        image = validate_image(file)

        # Save temporarily to uploads
        target_image_path = save_image(image, "target", UPLOADS_DIR)

        # Step 2: YOLO detect bottles
        bottles_results = []
        yolo_detections = []

        if yolo_detector:
            logger.info(f"[{request_id}] Running YOLO detection...")
            yolo_start = time.time()
            yolo_detections = yolo_detector.detect(target_image_path)
            yolo_elapsed = time.time() - yolo_start
            logger.info(f"[{request_id}] YOLO detected {len(yolo_detections)} bottles in {yolo_elapsed:.2f}s")

        if not llm_service:
            logger.warning(f"[{request_id}] LLM not configured")
            raise HTTPException(status_code=503, detail="LLM service not configured")

        # Step 3: If YOLO found bottles, recognize each; otherwise recognize full image
        if yolo_detections:
            for i, det in enumerate(yolo_detections):
                # Crop bottle region
                crop_img = yolo_detector.crop_bottle(target_image_path, det['bbox'], margin=10)
                from PIL import Image
                crop_pil = Image.fromarray(crop_img)

                # Save cropped image to crops directory
                crop_path = CROPS_DIR / f"crop_{request_id}_{i}.jpg"
                crop_pil.save(crop_path)
                crop_path_str = str(crop_path)

                # Call Qwen-VL
                logger.info(f"[{request_id}] Recognizing bottle {i+1}/{len(yolo_detections)}...")
                messages = [{
                    "role": "user",
                    "content": [
                        {"image": crop_path_str},
                        {"text": "请识别这个药瓶上的药品名称。只输出JSON格式: {\"name\":\"药品名称\", \"confidence\":0.0-1.0}。confidence表示识别置信度，0-1之间。只输出JSON，不要其他文字。"}
                    ]
                }]

                try:
                    import dashscope
                    from dashscope import MultiModalConversation

                    response = MultiModalConversation.call(
                        model=llm_service._model,
                        messages=messages,
                    )

                    if response.status_code == 200:
                        result = llm_service._parse_response(response)
                        medicine_info = result.get("medicine", {})

                        # 排除未识别到名称的药瓶
                        if medicine_info.get("name"):
                            bottles_results.append({
                                "name": medicine_info.get("name", ""),
                                "confidence": float(medicine_info.get("confidence", 0)) if medicine_info.get("confidence") else 0,
                                "bbox": det['bbox'],
                                "yolo_confidence": det['confidence'],
                            })
                    else:
                        logger.warning(f"Bottle {i+1} LLM error: {response.message}")

                except Exception as e:
                    logger.warning(f"Bottle {i+1} recognition failed: {e}")

                # Clean up crop file
                if os.path.exists(crop_path_str):
                    os.remove(crop_path_str)
        else:
            # No YOLO, recognize full image directly
            logger.info(f"[{request_id}] No YOLO detections, recognizing full image...")
            messages = [{
                "role": "user",
                "content": [
                    {"image": target_image_path},
                    {"text": "请识别这个药瓶上的药品名称。只输出JSON格式: {\"name\":\"药品名称\", \"confidence\":0.0-1.0}。confidence表示识别置信度，0-1之间。只输出JSON，不要其他文字。"}
                ]
            }]

            try:
                import dashscope
                from dashscope import MultiModalConversation

                response = MultiModalConversation.call(
                    model=llm_service._model,
                    messages=messages,
                )

                if response.status_code == 200:
                    result = llm_service._parse_response(response)
                    medicine_info = result.get("medicine", {})

                    # 排除未识别到名称的结果
                    if medicine_info.get("name"):
                        bottles_results.append({
                            "name": medicine_info.get("name", ""),
                            "confidence": float(medicine_info.get("confidence", 0)) if medicine_info.get("confidence") else 0,
                            "bbox": None,
                            "yolo_confidence": None,
                        })

            except Exception as e:
                logger.error(f"LLM call failed: {e}")

        total_elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Recognition done in {total_elapsed:.2f}s, found {len(bottles_results)} bottles")

        return RecognizeResponse(
            success=True,
            message=f"识别成功，发现 {len(bottles_results)} 个药瓶",
            medicine=MedicineInfo(
                name=bottles_results[0].get("name", "") if bottles_results else "",
                confidence=bottles_results[0].get("confidence", 0) if bottles_results else None,
            ) if bottles_results else None,
            bottles=bottles_results if bottles_results else None,
            matched_samples=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Stats API ====================


@app.get("/api/v1/stats")
async def stats():
    """Get statistics."""
    try:
        count = milvus_helper.get_total_count()

        return {
            "success": True,
            "total_samples": count,
            "collection": milvus_helper.get_collection_name(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)