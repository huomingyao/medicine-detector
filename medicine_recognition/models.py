"""
Pydantic models for the API.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class SampleCreate(BaseModel):
    """Model for creating a sample."""

    text_label: str = Field(..., description="Medicine label text")


class SampleResponse(BaseModel):
    """Model for sample response."""

    id: int
    image_path: str
    text_label: str


class SampleListResponse(BaseModel):
    """Model for sample list response."""

    samples: List[SampleResponse]
    total: int


class RecognizeRequest(BaseModel):
    """Model for recognize request."""

    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="Image URL")


class MedicineInfo(BaseModel):
    """Medicine information model."""

    name: str = Field(default="", description="Medicine name")
    confidence: Optional[float] = Field(default=None, description="Confidence score 0-1")
    specification: str = Field(default="", description="Specification")
    manufacturer: str = Field(default="", description="Manufacturer")
    usage: str = Field(default="", description="Usage instructions")
    warning: str = Field(default="", description="Warning")


class MatchedSample(BaseModel):
    """Matched sample model."""

    image_path: str
    label: str
    similarity: float


class DetectionResult(BaseModel):
    """YOLO detection result."""

    class_name: str
    confidence: float
    bbox: tuple


class RecognizeResponse(BaseModel):
    """Model for recognize response."""

    success: bool
    message: Optional[str] = None
    medicine: Optional[MedicineInfo] = None
    matched_samples: Optional[List[MatchedSample]] = None
    detections: Optional[List[DetectionResult]] = None
    bottles: Optional[List[Any]] = None  # Multiple bottles with name and confidence
    raw_response: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    message: str
    error_code: Optional[str] = None