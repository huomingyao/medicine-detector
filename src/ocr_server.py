
"""
OCR服务端程序 - 适配青岛大学附属医院输液单格式
使用GLM-OCR模型进行文字识别
支持批量处理图片

程序结构:
  1. 配置与依赖导入
  2. OCRRecognizer 类 - 模型加载与单张/批量图片识别
  3. 文本解析与信息抽取 - 患者信息、药品信息抽取规则
  4. 辅助函数 - 图片文件查找
  5. 主程序入口 - 命令行参数、批量处理与结果保存
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast
from dataclasses import dataclass
from PIL import Image
import numpy as np
import tqdm as tqdm_module
from tqdm import tqdm
import importlib
import re
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
import hashlib
import base64
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import lru_cache
from collections.abc import Iterator
import gc
import subprocess

_resample_target = getattr(Image, 'Resampling', Image)
RESAMPLE_BICUBIC = getattr(_resample_target, 'BICUBIC', getattr(Image, 'BICUBIC', 3))


def ensure_transformers_installed() -> None:
    try:
        import transformers  # type: ignore[import]
    except Exception as e:
        print('检测到 transformers 依赖不可用，尝试自动安装/修复。')
        print(f'  错误信息: {e}')
        print('  使用当前 Python 解释器执行 pip 安装。')
        try:
            subprocess.check_call([
                sys.executable,
                '-m',
                'pip',
                'install',
                '--upgrade',
                'transformers',
                'accelerate',
                'safetensors'
            ])
            import importlib
            importlib.invalidate_caches()
            import transformers  # type: ignore[import]
            print('transformers 依赖安装/修复完成。')
        except Exception as install_error:
            print('自动安装 transformers 失败，请手动执行以下命令：')
            print(f'  {sys.executable} -m pip install transformers accelerate safetensors')
            raise install_error


def import_torch() -> Any:
    try:
        import torch
        return torch
    except Exception as e:
        raise ImportError(f'无法导入 PyTorch: {e}') from e


def auto_zoom_image(image: Image.Image, min_content_side: int = 1200, max_side: int = 2048) -> Image.Image:
    """
    自动检测文档区域并放大目标区域。
    远拍图像中，如果文档内容区域较小而原图分辨率足够高，则优先裁剪并放大文档区域，提升 OCR 识别效果。
    """
    width, height = image.size
    if max(width, height) < min_content_side:
        return image

    try:
        bbox = find_document_bbox(image)
    except Exception:
        bbox = None

    if bbox is not None:
        left, top, right, bottom = bbox
        doc_w = right - left
        doc_h = bottom - top
        if doc_w <= 0 or doc_h <= 0:
            return image

        content_ratio = max(doc_w / width, doc_h / height)
        needs_zoom = content_ratio < 0.75 or max(doc_w, doc_h) < min_content_side
        if needs_zoom:
            cropped = image.crop((left, top, right, bottom))
            scale = max(1.0, min_content_side / max(doc_w, doc_h))
            max_scale = max_side / max(doc_w, doc_h)
            scale = min(scale, max_scale)
            if scale > 1.0:
                new_size = (min(int(doc_w * scale), max_side), min(int(doc_h * scale), max_side))
                return cropped.resize(new_size, resample=RESAMPLE_BICUBIC)
            return cropped

    # 如果未能定位文档区域，但图片本身仍然较小且分辨率充足，则尝试整体放大
    if min(width, height) < min_content_side and max(width, height) >= min_content_side:
        scale = min(max_side / max(width, height), min_content_side / min(width, height))
        if scale > 1.0:
            new_size = (min(int(width * scale), max_side), min(int(height * scale), max_side))
            return image.resize(new_size, resample=RESAMPLE_BICUBIC)

    return image


def prepare_image_for_recognition(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """
    如果图片分辨率过大，按比例缩小到 max_side，避免显存爆炸。
    同时对远拍小目标图片做自动放大处理，提高 OCR 识别准确率。
    """
    image = auto_zoom_image(image)
    width, height = image.size
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, resample=RESAMPLE_BICUBIC)
    return image


def preprocess_image_for_ocr(image: Image.Image, max_side: int = 3840) -> Image.Image:
    """
    统一的 OCR 预处理入口，用于 HTTP 服务和批量处理中的图像预处理。
    """
    image = auto_correct_image(image)
    return prepare_image_for_recognition(image, max_side=max_side)


def _horizontal_edge_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert('L'), dtype=np.int16)
    if gray.shape[0] < 2:
        return 0.0
    gradient = np.abs(np.diff(gray, axis=0))
    return float(np.mean(gradient))


def detect_best_quadrant_rotation(image: Image.Image) -> int:
    """通过候选 0/90/180/270 度旋转选择最可能的横向文字方向。"""
    best_angle = 0
    best_score = -1.0
    for angle in (0, 90, 180, 270):
        rotated = image.rotate(angle, expand=True)
        score = _horizontal_edge_score(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def estimate_skew_angle(image: Image.Image) -> float:
    """使用 OpenCV 检测图像轻微倾斜角度，返回需要旋转的角度。"""
    try:
        import cv2
    except ImportError:
        return 0.0

    gray = np.asarray(image.convert('L'))
    if gray.ndim != 2:
        return 0.0

    h, w = gray.shape
    if h < 20 or w < 20:
        return 0.0

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    if edges.sum() == 0:
        return 0.0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(100, min(w, h) // 4),
        minLineLength=min(w, h) // 4,
        maxLineGap=20
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def auto_correct_image(image: Image.Image) -> Image.Image:
    """自动校正图片方向（旋转、翻转）并做轻微倾斜纠正。"""
    image = image.convert('RGB')
    correction_angle = 0

    # 使用四角旋转检测
    correction_angle = detect_best_quadrant_rotation(image)

    if correction_angle != 0:
        image = image.rotate(-correction_angle, expand=True, fillcolor=(255, 255, 255))

    skew_angle = estimate_skew_angle(image)
    if abs(skew_angle) > 0.5:
        image = image.rotate(-skew_angle, expand=True, fillcolor=(255, 255, 255))
    return image


def crop_patient_block(image: Image.Image, left_ratio: float = 0.05, top_ratio: float = 0.10, width_ratio: float = 0.55, height_ratio: float = 0.18) -> Image.Image:
    """
    裁剪出固定位置的患者信息区域。
    该区域为上部左侧，尽量只包含患者姓名/年龄/性别行，排除右侧时间和下方药品区。
    """
    width, height = image.size
    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = int(width * width_ratio)
    bottom = int(height * (top_ratio + height_ratio))
    return image.crop((left, top, right, bottom))


def crop_patient_name_block(image: Image.Image, left_ratio: float = 0.05, top_ratio: float = 0.18, width_ratio: float = 0.45, height_ratio: float = 0.12) -> Image.Image:
    """
    裁剪出更窄的患者姓名区域，仅识别姓名行，用于严格提取患者姓名。
    """
    width, height = image.size
    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = int(width * width_ratio)
    bottom = int(height * (top_ratio + height_ratio))
    return image.crop((left, top, right, bottom))


def find_document_bbox(image: Image.Image) -> Optional[tuple[int, int, int, int]]:
    gray = image.convert('L')
    arr = np.asarray(gray)
    try:
        cv2 = importlib.import_module('cv2')  # type: ignore[import]
        blur = cv2.GaussianBlur(arr, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        for cnt in contours[:3]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue
            if w < arr.shape[1] * 0.3 or h < arr.shape[0] * 0.3:
                continue
            return x, y, x + w, y + h
    except Exception:
        pass

    mask = arr > 180
    if mask.sum() < 100:
        return None
    ys, xs = np.where(mask)
    top, left = int(ys.min()), int(xs.min())
    bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
    if right - left < 50 or bottom - top < 50:
        return None
    return left, top, right, bottom


def find_top_right_digit_block(image: Image.Image) -> Image.Image:
    """
    在图片右上角查找可能的操作员ID数字块。
    """
    doc_bbox = find_document_bbox(image)
    if doc_bbox is not None:
        doc_left, doc_top, doc_right, doc_bottom = doc_bbox
        doc_w = doc_right - doc_left
        doc_h = doc_bottom - doc_top
        search_left = doc_left + int(doc_w * 0.25)
        search_top = doc_top
        search_right = min(doc_right, doc_left + int(doc_w * 0.98))
        search_bottom = min(doc_top + int(doc_h * 0.40), doc_bottom)
    else:
        width, height = image.size
        search_left = int(width * 0.25)
        search_top = 0
        search_right = min(width, int(width * 0.98))
        search_bottom = min(height, int(height * 0.40))
    region = image.crop((search_left, search_top, search_right, search_bottom))

    try:
        cv2 = importlib.import_module('cv2')  # type: ignore[import]
        gray = np.asarray(region.convert('L'))
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = 255 - bw
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < region.width * 0.12 or h < region.height * 0.18:
                continue
            if w > region.width * 0.9 or h > region.height * 0.9:
                continue
            ratio = w / max(1.0, h)
            if ratio < 0.25 or ratio > 4.0:
                continue
            candidates.append((x, y, w, h))

        if candidates:
            candidates.sort(key=lambda item: (item[2] * item[3], item[0]), reverse=True)
            x, y, w, h = candidates[0]
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(region.width, x + w + pad)
            y1 = min(region.height, y + h + pad)
            cropped = region.crop((x0, y0, x1, y1))
            if cropped.size[0] >= 24 and cropped.size[1] >= 24:
                return cropped
    except Exception:
        pass

    return region


def crop_operator_id_block(image: Image.Image) -> Image.Image:
    """
    裁剪出一个默认的顶部右侧操作员ID区域，作为主OCR候选块。
    """
    return find_top_right_digit_block(image)


def recognize_digits_with_tesseract(image: Image.Image) -> str:
    try:
        pytesseract = importlib.import_module('pytesseract')  # type: ignore[import]
    except Exception:
        return ''

    try:
        gray = image.convert('L')
        bw = gray.point(lambda x: 0 if int(x) < 160 else 255, '1')
        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        best_num = ''
        best_score = 0
        best_left = -1
        best_top = 10**9
        for i, text in enumerate(data['text']):
            if not text:
                continue
            text = text.strip()
            if not re.fullmatch(r'\d{2,6}', text):
                continue
            height = int(data['height'][i] or 0)
            left = int(data['left'][i] or 0)
            top = int(data['top'][i] or 0)
            score = height * len(text)
            if (
                top < best_top
                or (top == best_top and score > best_score)
                or (top == best_top and score == best_score and left > best_left)
            ):
                best_num = text
                best_score = score
                best_left = left
                best_top = top
        if best_num:
            return best_num

        text = pytesseract.image_to_string(bw, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        candidates = re.findall(r'\b(\d{2,3})\b', text)
        if candidates:
            return candidates[0]
        return ''
    except Exception:
        return ''


def recognize_operator_id_with_tesseract(image: Image.Image) -> str:
    return recognize_digits_with_tesseract(image)


def recognize_top_right_number_with_tesseract(image: Image.Image) -> str:
    try:
        pytesseract = importlib.import_module('pytesseract')  # type: ignore[import]
    except Exception:
        return ''

    cropped = find_top_right_digit_block(image)
    try:
        gray = cropped.convert('L')
        bw = gray.point(lambda x: 0 if int(x) < 160 else 255, '1')
        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config='--psm 6 -c tessedit_char_whitelist=0123456789')
    except Exception:
        return ''

    best_num = ''
    best_score = -1
    for i, text in enumerate(data.get('text', [])):
        if not text:
            continue
        digits = re.sub(r'[^0-9]', '', text)
        if not re.fullmatch(r'\d{2,3}', digits):
            continue
        conf_text = data.get('conf', [])
        width_text = data.get('width', [])
        height_text = data.get('height', [])
        conf = int(conf_text[i]) if i < len(conf_text) and str(conf_text[i]).isdigit() else 0
        width = int(width_text[i]) if i < len(width_text) and str(width_text[i]).isdigit() else 0
        height = int(height_text[i]) if i < len(height_text) and str(height_text[i]).isdigit() else 0
        score = conf * max(1, width * height)
        if score > best_score:
            best_score = score
            best_num = digits
    return best_num


def recognize_with_pytesseract(image: Image.Image) -> str:
    try:
        pytesseract = importlib.import_module('pytesseract')  # type: ignore[import]
    except Exception:
        print('pytesseract 未安装，无法使用快速模式 OCR。')
        return ''

    try:
        gray = image.convert('L')
        bw = gray.point(lambda x: 0 if int(x) < 150 else 255, '1')
        text = pytesseract.image_to_string(bw, config='--psm 6')
        return text.strip()
    except Exception as e:
        print(f'pytesseract OCR 失败：{e}')
        return ''


def extract_operator_id_from_text(text: str, crop: bool = False) -> str:
    if not text:
        return ''

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if crop:
        lines = [line for line in lines if not re.search(r'[A-Za-z\u4e00-\u9fa5]', line)]
        for line in lines:
            if re.fullmatch(r'\d{2,3}', line):
                return line
        candidates = []
        for line in lines:
            nums = re.findall(r'\d{2,3}', line)
            for num in nums:
                if num not in {'10', '12', '16', '25', '100'}:
                    candidates.append(num)
        if candidates:
            return candidates[0]

    patterns = [
        r'操作员(?:ID|号)?[：:]?\s*([A-Za-z0-9\-]{2,10})',
        r'操作员[：:]?\s*([A-Za-z0-9\-]{2,10})',
        r'操作员ID[：:]?\s*([A-Za-z0-9\-]{2,10})',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()

    if crop:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate_patterns = [
            r'\b[A-Za-z0-9\-]{2,10}\b',
            r'\b\d{2,6}\b'
        ]
        for line in lines:
            if re.search(r'[一-龥]', line):
                continue
            for pat in candidate_patterns:
                for match in re.finditer(pat, line):
                    value = match.group(0).strip()
                    if re.search(r'\d', value):
                        return value
        return ''

    # 全文回退：优先匹配包含数字的短 ID
    strong_candidates = re.findall(r'\b[A-Za-z0-9\-]{2,10}\b', text)
    for cand in strong_candidates:
        if re.search(r'\d', cand) and cand not in {'10', '12', '16', '25', '100'}:
            return cand

    all_numbers = re.findall(r'\b(\d{2,6})\b', text)
    for num in all_numbers:
        if num not in {'10', '12', '16', '25', '100'}:
            return num
    if all_numbers:
        return all_numbers[0]
    return ''


def _rgb_from_hue(h: float) -> tuple[int, int, int]:
    h = h % 360
    c = 1
    x = 1 - abs((h / 60) % 2 - 1)
    if h < 60:
        r, g, b = 1, x, 0
    elif h < 120:
        r, g, b = x, 1, 0
    elif h < 180:
        r, g, b = 0, 1, x
    elif h < 240:
        r, g, b = 0, x, 1
    elif h < 300:
        r, g, b = x, 0, 1
    else:
        r, g, b = 1, 0, x
    return int(r * 255), int(g * 255), int(b * 255)


def _gradient_ansi_colors(steps: int = 100) -> List[str]:
    colours: List[str] = []
    for i in range(steps):
        hue = i * 300 / max(1, steps - 1)
        r, g, b = _rgb_from_hue(hue)
        colours.append(f'\x1b[38;2;{r};{g};{b}m')
    return colours

_GRADIENT_COLOURS = _gradient_ansi_colors(100)


# === 2. OCR识别器类 ===
# 负责模型加载、单张图片识别和批量识别
class OCRRecognizer:
    """OCR识别器类，使用单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化OCR识别器
        使用单例模式确保模型只加载一次
        """
        if not OCRRecognizer._initialized:
            self.LOCAL_MODEL_PATH = r"D:/GLM-OCR"
            self.DEVICE = "cpu"
            self.MAX_NEW_TOKENS = 1024
            self.BATCH_SIZE = 1
            self.MODEL_NAME = self.LOCAL_MODEL_PATH
            self.TOKENIZER_NAME = self.LOCAL_MODEL_PATH
            self.model: Any = None
            self.processor: Any = None
            self.load_model()
            OCRRecognizer._initialized = True

    def load_model(self):
        print("加载OCR模型...")
        print("  [0/4] 导入 transformers 库...")
        start_import_time = time.time()
        try:
            from transformers import (
                AutoProcessor,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoModelForSeq2SeqLM,
            )
            torch = import_torch()
        except KeyboardInterrupt:
            print('      Transformers 导入被中断，请稍后重试或检查 Python 环境。')
            raise
        except Exception as e:
            print(f'      Transformers 导入失败: {e}')
            print('      请确认运行脚本的 Python 环境中已安装 transformers。')
            print('      如果使用 Windows Store Python，请切换到虚拟环境或 Anaconda，并执行:')
            print('          python -m pip install transformers accelerate safetensors')
            print(f'      当前 Python 解释器: {sys.executable}')
            raise
        print(f"      Transformers 导入完成 ({time.time() - start_import_time:.1f}s)")

        print("  [1/4] 加载处理器...")
        start_time = time.time()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.TOKENIZER_NAME,
                trust_remote_code=True
            )
            print(f"      完成 ({time.time() - start_time:.1f}s)")
        except Exception as e:
            print(f"      处理器加载失败: {e}")
            raise

        print("  [2/3] 加载模型权重...")
        model_start_time = time.time()
        use_dtype = torch.float16 if self.DEVICE == "cuda" else torch.float32
        start_time = time.time()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.TOKENIZER_NAME,
                trust_remote_code=True
            )
            print(f"      完成 ({time.time() - start_time:.1f}s)")
        except Exception as e:
            print(f"      处理器加载失败: {e}")
            raise
        
        print("  [2/3] 加载模型权重...")
        model_start_time = time.time()
        
        use_dtype = torch.float16 if self.DEVICE == "cuda" else torch.float32
        
        try:
            from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                dtype=use_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.DEVICE)
            print(f"      专用模型加载完成 ({time.time() - model_start_time:.1f}s)")
        except Exception as e:
            print(f"      专用模型加载失败: {e}")
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.MODEL_NAME,
                    dtype=use_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.DEVICE)
                print(f"      AutoModelForImageTextToText 加载完成 ({time.time() - model_start_time:.1f}s)")
            except Exception as e2:
                print(f"      AutoModelForImageTextToText 加载失败: {e2}")
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.MODEL_NAME,
                        dtype=use_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.to(self.DEVICE)
                    print(f"      AutoModelForSeq2SeqLM 加载完成 ({time.time() - model_start_time:.1f}s)")
                except Exception as e3:
                    print(f"      AutoModelForSeq2SeqLM 加载失败: {e3}")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.MODEL_NAME,
                            dtype=use_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        self.model = self.model.to(self.DEVICE)
                        print(f"      备选模型加载完成 ({time.time() - model_start_time:.1f}s)")
                    except Exception as e4:
                        print(f"      所有模型加载方式都失败: {e4}")
                        raise
        
        print("  [3/3] 模型优化...")
        if self.DEVICE == "cuda":
            self.model.eval()
            print("      GPU模式，模型已设置为评估模式")
        else:
            print("      CPU模式，跳过GPU优化")
        print("模型加载完成！")

    def recognize(self, image: Union[str, Image.Image], task_prompt: str = "") -> str:
        """
        识别单张图片
        Args:
            image: 图片路径或PIL Image对象
            task_prompt: 任务提示词（已废弃，GLM-OCR不需要）
        Returns:
            识别结果文本
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            image = preprocess_image_for_ocr(image)

            # GLM-OCR 使用 apply_chat_template 的方式处理输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "请识别图片中的文字。"}
                    ]
                }
            ]
            
            # 使用 processor.apply_chat_template 处理消息
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 将输入移到设备上
            torch = import_torch()
            inputs = {k: v.to(self.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            inputs = cast(Dict[str, Any], inputs)
            
            # 检查 input_ids 是否存在
            if 'input_ids' not in inputs or inputs['input_ids'] is None:
                print("OCR 处理器返回的 inputs 中没有 input_ids")
                return ""
            
            # 确保 input_ids 是 Long 类型
            if inputs['input_ids'].dtype != torch.long:
                inputs['input_ids'] = inputs['input_ids'].long()
            
            if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                if inputs['attention_mask'].dtype != torch.long:
                    inputs['attention_mask'] = inputs['attention_mask'].long()

            # 生成文本
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True
                    )
            except Exception as e:
                print(f"OCR 模型识别失败：{e}")
                raise
            
            # 解码结果
            input_ids_length = inputs["input_ids"].shape[1]
            output_text = self.processor.decode(
                generated_ids[0][input_ids_length:],
                skip_special_tokens=True
            )
            
            result = output_text.strip()
            return result if result else ""
        except KeyboardInterrupt:
            print("OCR 识别被用户中断。")
            raise
        except Exception as e:
            print(f"OCR 识别错误：{e}")
            import traceback
            traceback.print_exc()
            return ""

    def recognize_batch(self, images: List[Union[str, Image.Image]], task_prompt: str = "") -> List[str]:
        """
        批量识别图片
        Args:
            images: 图片列表
            task_prompt: 任务提示词
        Returns:
            识别结果列表
        """
        results = []
        for img in tqdm(images, desc="批量识别", unit="张"):
            result = self.recognize(img, task_prompt)
            results.append(result)
        return results
    
    def recognize_prescription(self, image_path: str, show_progress: bool = False) -> dict:
        """
        识别处方图片（兼容旧接口）
        Args:
            image_path: 图片路径
            show_progress: 是否显示进度（已废弃）
        Returns:
            识别结果字典
        """
        text = self.recognize(image_path)
        return {
            "full_text": text,
            "text": text
        }

# === 3. 文本解析与信息抽取 ===
# 包含病人信息提取、药品识别、文本清洗等规则

def split_prescription_sections(text: str) -> Dict[str, List[str]]:
    """
    基于输液单格式做区域定位，将识别结果分为：
      - patient: 患者信息区
      - medicine: 药品表格区
      - footer: 备注/签字等页脚区
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patient_lines: List[str] = []
    medicine_lines: List[str] = []
    footer_lines: List[str] = []
    in_medicine_section = False

    table_header_keywords = ['药品名称', '药品名', '药名', '品名']
    footer_keywords = ['备注', '说明', '医生', '审方', '打签', '配置', '复核', '第']

    for line in lines:
        if any(keyword in line for keyword in table_header_keywords) or ('剂量' in line and '规格' in line):
            in_medicine_section = True
            continue

        if in_medicine_section and any(keyword in line for keyword in footer_keywords):
            in_medicine_section = False
            footer_lines.append(line)
            continue

        if in_medicine_section:
            medicine_lines.append(line)
        else:
            if footer_lines:
                footer_lines.append(line)
            else:
                patient_lines.append(line)

    return {'patient': patient_lines, 'medicine': medicine_lines, 'footer': footer_lines}


def extract_medicine_info(text: str) -> List[Dict[str, str]]:
    """
    启发式药品信息提取
    使用关键词检测 + 上下文推断 + 智能拆分
    """
    medicines = []
    lines = text.split('\n')

    medicine_keywords = [
        '氯化钠', '葡萄糖', '氟比洛芬', '氰比洛芬', '头孢', '青霉素', '维生素',
        '注射液', '氨基酸', '脂肪乳', '钾', '钠', '钙', '镁',
        '羟乙基', '洛芬', '西林', '硝唑', '嘧啶', '洛尔', '氨', '素', '苷', '霉素'
    ]

    skip_keywords = ['备注', '说明', '医生', '审方', '打签', '配置', '复核', '药种',
                    '药品名称', '药名', '名称', '剂量', '规格', '用法', '频次', '时间',
                    '输液单', '处方', '门诊', '补-', '批', '贴', '数']

    def is_medicine_line(line: str) -> bool:
        for kw in medicine_keywords:
            if kw in line:
                return True
        return False

    def is_skip_line(line: str) -> bool:
        for kw in skip_keywords:
            if line.startswith(kw) or line == kw:
                return True
        return False

    def smart_split(line: str) -> List[str]:
        parts = []
        current = ""
        i = 0
        while i < len(line):
            c = line[i]
            if c in '（）()【】[]':
                if current:
                    parts.append(current)
                    current = ""
                parts.append(c)
            elif c == ' ' or c == '\t':
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += c
            i += 1
        if current:
            parts.append(current)
        return parts

    def parse_line(line: str) -> dict:
        result = {'名称': '', '剂量': '', '规格': ''}
        line = line.strip()
        if not line or is_skip_line(line):
            return result

        parts = smart_split(line)

        name_parts = []
        dosage_parts = []
        spec_parts = []
        reading = 'name'

        for part in parts:
            if not part or part in '（）()【】[]':
                continue
            if part in ['◇', '◆', '○', '△']:
                continue

            is_number = False
            try:
                float(part.replace('%', '').replace('μ', ''))
                is_number = True
            except:
                pass

            if is_number:
                if '%' in part or 'ml' in part.lower() or 'mg' in part.lower() or 'g' in part.lower():
                    if not dosage_parts:
                        dosage_parts.append(part)
                    else:
                        spec_parts.append(part)
                else:
                    if reading == 'name' and name_parts:
                        dosage_parts.append(part)
                    else:
                        dosage_parts.append(part)
            else:
                name_parts.append(part)
                reading = 'name'

        result['名称'] = ''.join(name_parts)
        result['剂量'] = ' '.join(dosage_parts) if dosage_parts else ''
        result['规格'] = ' '.join(spec_parts) if spec_parts else ''

        return result

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_skip_line(line):
            continue
        if is_medicine_line(line):
            med = parse_line(line)
            if med['名称'] and len(med['名称']) > 1:
                medicines.append(med)

    seen = set()
    unique = []
    for med in medicines:
        key = med.get('名称', '')
        if key not in seen:
            seen.add(key)
            unique.append(med)
    return unique


def is_valid_patient_name(name: str) -> bool:
    return bool(re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', name))


def check_and_fix_patient_info(patient_info: Dict[str, str], section_text: str, full_text: str, fallback_name: Optional[str] = None) -> tuple[Dict[str, str], List[str]]:
    warnings: List[str] = []
    name = patient_info.get('姓名', '')
    if not name or not re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', name):
        fallback = extract_patient_info(full_text)
        if fallback.get('姓名') and re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', fallback['姓名']):
            patient_info = fallback
            warnings.append('患者姓名校验失败，已尝试全文回退提取')
        elif fallback_name and re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', fallback_name):
            patient_info['姓名'] = fallback_name
            warnings.append('患者姓名缺失，已使用上一张有效患者姓名')
        else:
            warnings.append('患者姓名校验失败，未找到有效姓名')

    age = patient_info.get('年龄', '')
    if age and not re.fullmatch(r'\d{1,3}岁', age):
        warnings.append('年龄格式异常')

    gender = patient_info.get('性别', '')
    if gender and gender not in ['男', '女']:
        warnings.append('性别格式异常')

    if not patient_info.get('医院'):
        warnings.append('医院名称缺失')

    return patient_info, warnings


def check_and_fix_medicine_info(medicines: List[Dict[str, str]], section_text: str, full_text: str) -> tuple[List[Dict[str, str]], List[str]]:
    warnings: List[str] = []

    invalid_names = {'药品名称', '剂量', '规格'}

    def valid_med_item(item: Dict[str, str]) -> bool:
        name = item.get('名称', '')
        if not name or len(name) < 2 or name in invalid_names:
            return False
        return True

    medicines = [med for med in medicines if valid_med_item(med)]

    if not medicines or any(not valid_med_item(item) for item in medicines):
        fallback = extract_medicine_info(full_text)
        if fallback and len(fallback) >= len(medicines):
            medicines = fallback
            warnings.append('药品条目校验失败，已尝试全文回退提取')
        elif not medicines:
            warnings.append('未能提取到药品信息')
        else:
            warnings.append('药品条目存在不完整项')

    return medicines, warnings


def extract_patient_region(text: str) -> List[str]:
    """
    识别患者区域：从顶部开始到药品表头之前的区域。
    这个区域内的文本才是患者信息，超出该区域就不是患者信息。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patient_region: List[str] = []
    header_keywords = ['药品名称', '药品名', '药名', '品名', '剂量', '规格']
    for line in lines:
        if any(keyword in line for keyword in header_keywords):
            break
        patient_region.append(line)
    return patient_region


def extract_patient_name_from_text(text: str) -> Optional[str]:
    invalid_names = {
        '医院', '科室', '病区', '病房', '处方', '门诊', '医生', '长期', '静脉', '滴注', '注射',
        '规格', '剂量', '药品', '药名', '药品名称', '药物', '备注', '说明', '备注说明', '配置', '审方', '打签', '复核'
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if re.search(r'\d+岁', line):
            match = re.search(r'([\u4e00-\u9fa5]{2,4})\s+\d+岁', line)
            if match:
                name = match.group(1).strip()
                if name not in invalid_names:
                    return name

            if idx > 0:
                candidate = lines[idx - 1]
                if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', candidate) and candidate not in invalid_names:
                    return candidate

            if idx + 1 < len(lines):
                candidate = lines[idx + 1]
                if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', candidate) and candidate not in invalid_names:
                    return candidate

    match = re.search(r'(?:姓名|患者)[：:]\s*([\u4e00-\u9fa5]{2,4})', text)
    if match:
        name = match.group(1).strip()
        if name not in invalid_names:
            return name

    return None


def extract_patient_info(text: str) -> Dict[str, str]:
    """
    从OCR文本中提取患者信息（适配青岛大学附属医院输液单格式）
    Args:
        text: OCR识别结果文本
    Returns:
        患者信息字典
    """
    patient_info = {}
    patient_region_lines = extract_patient_region(text)
    region_text = '\n'.join(patient_region_lines)

    # 提取医院名称
    hospital_match = re.search(r'(青岛大学附属医院|[^\n]+医院)', region_text)
    if hospital_match:
        patient_info['医院'] = hospital_match.group(1).strip()
    
    # 提取科室/病区
    dept_match = re.search(r'([^\n]+(?:科|病区|病房))', region_text)
    if dept_match:
        patient_info['科室'] = dept_match.group(1).strip()
    
    # 提取患者姓名 - 多种模式尝试
    invalid_names = {
        '医院', '科室', '病区', '病房', '处方', '门诊', '医生', '长期', '静脉', '滴注', '注射', '规格', '剂量', '药品', '药名',
        '药品名称', '备注', '说明', '备注说明', '配置', '审方', '打签', '复核'
    }
    name_patterns = [
        r'(?:姓名|患者)[：:]\s*([\u4e00-\u9fa5]{2,4})',
        r'([\u4e00-\u9fa5]{2,4})\s+\d+岁',
        r'(?:科|病区|病房)\s+([\u4e00-\u9fa5]{2,4})\s+\d+岁',
        r'([\u4e00-\u9fa5]{2,4})\s+\d{1,2}[:：]',
        r'\n([\u4e00-\u9fa5]{2,4})\s+(?:岁|女|男)',
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, region_text)
        if name_match:
            name = name_match.group(1).strip()
            if name not in invalid_names:
                patient_info['姓名'] = name
                break

    if '姓名' not in patient_info:
        for idx, line in enumerate(patient_region_lines[:-1]):
            if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', line) and not re.search(r'配|药|品|注射|输液|剂量|规格|药名|药品|备注|说明|医院|科室|病区|病房', line):
                next_line = patient_region_lines[idx + 1]
                if re.search(r'\d+岁', next_line) and re.search(r'[男女]', next_line):
                    patient_info['姓名'] = line
                    break

    if '姓名' not in patient_info:
        for idx, line in enumerate(patient_region_lines[:-1]):
            if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', line) and line not in invalid_names:
                if idx + 1 < len(patient_region_lines) and re.search(r'\d+岁|男|女', patient_region_lines[idx + 1]):
                    patient_info['姓名'] = line
                    break

    if '姓名' not in patient_info:
        for line in patient_region_lines:
            only_name = re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', line)
            if only_name and line not in invalid_names and not re.search(r'备注|说明|配置|审方|打签|复核|药|品|注射|输液|剂量|规格|医院|科室|病区|病房|门诊', line):
                patient_info['姓名'] = line
                break

    if '姓名' not in patient_info:
        full_text_name = extract_patient_name_from_text(text)
        if full_text_name:
            patient_info['姓名'] = full_text_name
    rx_match = re.search(r'(?:处方|门诊)[：:]\s*([A-Za-z0-9]+)', text)
    if rx_match:
        patient_info['处方号'] = rx_match.group(1).strip()

    # 提取年龄
    age_match = re.search(r'(\d+)岁', text)
    if age_match:
        patient_info['年龄'] = age_match.group(1) + '岁'
    
    # 提取性别
    gender_match = re.search(r'([男女])', text)
    if gender_match:
        patient_info['性别'] = gender_match.group(1)
    
    # 提取手机号/电话
    phone_match = re.search(r'(?:电话|手机号|手机)[：:]?\s*([0-9\- ]{6,20})', text)
    if phone_match:
        patient_info['手机号'] = phone_match.group(1).strip()

    # 提取住院号/病历号
    id_match = re.search(r'(?:住院号|病历号)[：:]?\s*([0-9]{5,12})', text)
    if id_match:
        patient_info['住院号'] = id_match.group(1)
    else:
        generic_id_match = re.search(r'\b([0-9]{7,12})\b', text)
        if generic_id_match:
            patient_info['住院号'] = generic_id_match.group(1)

    # 提取主治医师
    doctor_match = re.search(r'(?:主治医师|主治医生|责任医师|医生)[：:]?\s*([\u4e00-\u9fa5]{2,4})', text)
    if doctor_match:
        patient_info['主治医师'] = doctor_match.group(1).strip()

    # 统一号码字段
    if patient_info.get('手机号'):
        patient_info['号码'] = patient_info['手机号']
    elif patient_info.get('住院号'):
        patient_info['号码'] = patient_info['住院号']
    
    # 提取床号
    bed_match = re.search(r'床号[：:]?\s*(\d+)', text)
    if bed_match:
        patient_info['床号'] = bed_match.group(1)
    else:
        # 尝试匹配输液单上的床号格式
        bed_match2 = re.search(r'\b(\d{2,3})\s*床', text)
        if bed_match2:
            patient_info['床号'] = bed_match2.group(1)
    
    # 提取医嘱类型
    order_match = re.search(r'(静脉滴注|静脉注射|口服|皮下注射|肌内注射)', text)
    if order_match:
        patient_info['医嘱类型'] = order_match.group(1)
    
    # 提取频次
    freq_match = re.search(r'(Bid|Qd|Tid|Qid|Q12h|Q8h|Q6h|长期|临时)', text, re.IGNORECASE)
    if freq_match:
        patient_info['频次'] = freq_match.group(1)
    
    # 提取时间
    time_match = re.search(r'(\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})', text)
    if time_match:
        patient_info['时间'] = time_match.group(1)
    
    return patient_info


# === 4. 辅助函数 ===
# 包括图片查找、路径处理等通用工具

def find_images_in_directory(directory: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) -> List[str]:
    """
    在目录中查找图片文件
    Args:
        directory: 目录路径
        extensions: 图片文件扩展名
    Returns:
        图片文件路径列表
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def format_result_text(all_results: Dict[str, Any]) -> str:
    lines: List[str] = []
    for img_path, result in all_results.items():
        lines.append(f"===== {os.path.basename(img_path)} =====")
        if 'error' in result:
            lines.append(f"错误：{result['error']}")
            lines.append("")
            continue

        patient_info = result.get('patient_info', {})
        lines.append(f"患者：{patient_info.get('姓名', '')}")
        lines.append(f"医院：{patient_info.get('医院', '')}")
        lines.append(f"号码：{patient_info.get('号码', '')}")
        lines.append(f"主治医师：{patient_info.get('主治医师', '')}")

        extra_keys = ['科室', '年龄', '性别', '住院号', '床号', '处方号', '医嘱类型', '频次', '时间']
        for key in extra_keys:
            if key in patient_info and key not in {'姓名', '医院'}:
                lines.append(f"{key}：{patient_info[key]}")

        medicines = result.get('medicines', [])
        if medicines:
            lines.append("药品信息：")
            for idx, med in enumerate(medicines, 1):
                med_fields = '；'.join(f"{k}:{v}" for k, v in med.items() if v)
                lines.append(f"  {idx}. {med_fields}")

        warnings = result.get('warnings', [])
        if warnings:
            lines.append("警告：")
            for warn in warnings:
                lines.append(f"  - {warn}")

        lines.append("")
    return "\n".join(lines)


def extract_top_section_info(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    top_info: Dict[str, str] = {}
    patient_region_lines = extract_patient_region(text)

    if not patient_region_lines:
        return top_info

    header_lines = patient_region_lines[:8]

    for line in header_lines:
        if '输液单' in line or '补-' in line or '批' in line or '药单' in line:
            top_info['药单名字'] = line
            break
    if '药单名字' not in top_info and header_lines:
        top_info['药单名字'] = header_lines[0]

    for line in header_lines[1:6]:
        if '医院' in line or '病区' in line or '科' in line:
            top_info['医院'] = line
            break

    for line in header_lines[1:7]:
        if any(keyword in line for keyword in ['静脉滴注', '静脉注射', '口服', '皮下注射', '肌内注射', 'Bid', 'Qd', 'Tid', 'Qid', '长期', '临时']):
            top_info['疗程名字'] = line
            break

    date_time_match = re.search(r'\b(\d{1,2}[ /-]\d{1,2}\s+\d{1,2}:\d{2})\b', text)
    if not date_time_match:
        date_time_match = re.search(r'\b(\d{1,2}\s+\d{1,2}\s+\d{1,2}:\d{2})\b', text)
    if date_time_match:
        top_info['日期时间'] = date_time_match.group(1)

    name_line = ''
    for idx, line in enumerate(header_lines):
        if re.fullmatch(r'[一-龥]{2,4}', line):
            if not re.search(r'药|品|注射|输液|剂量|规格|医院|科室|病区|病房|门诊|备注|说明|审方|打签|复核', line):
                name_line = line
                break
        if re.fullmatch(r'[一-龥]{2,4}', line):
            name_line = line
            break
    if not name_line:
        for line in header_lines:
            match = re.search(r'(?:姓名|患者)[：:]?\s*([一-龥]{2,4})', line)
            if match:
                name_line = match.group(1).strip()
                break
    if name_line:
        top_info['患者'] = name_line

    age_line = ''
    for line in header_lines:
        if re.search(r'\d{1,3}岁', line) and re.search(r'[男女]', line):
            age_line = line
            break
    if age_line:
        age_match = re.search(r'(\d{1,3})岁', age_line)
        if age_match:
            top_info['年龄'] = age_match.group(1) + '岁'
        gender_match = re.search(r'(男|女)', age_line)
        if gender_match:
            top_info['性别'] = gender_match.group(1)
        id_matches = re.findall(r'\b\d{5,12}\b', age_line)
        if id_matches:
            for value in id_matches:
                if value != age_match.group(1):
                    top_info['号码'] = value
                    break

    return top_info


def extract_operator_id(text: str) -> str:
    if not text:
        return ''
    patterns = [
        r'操作员(?:ID|号)?[：:]?\s*([A-Za-z0-9\-]+)',
        r'操作员[：:]?\s*([A-Za-z0-9\-]+)',
        r'操作员ID[：:]?\s*([A-Za-z0-9\-]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()
    return ''


def build_structured_result(image_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    patient_info = result.get('patient_info', {})
    top_info = extract_top_section_info(result.get('full_text', ''))

    structured: Dict[str, Any] = {
        '图片文件': os.path.basename(image_path),
        '药单名字': top_info.get('药单名字', ''),
        '医院': top_info.get('医院', patient_info.get('医院', '')),
        '疗程名字': top_info.get('疗程名字', ''),
        '日期时间': top_info.get('日期时间', ''),
        '患者': top_info.get('患者', patient_info.get('姓名', '')),
        '年龄': top_info.get('年龄', patient_info.get('年龄', '')),
        '性别': top_info.get('性别', patient_info.get('性别', '')),
        '号码': top_info.get('号码', patient_info.get('号码', '')),
        '主治医师': patient_info.get('主治医师', ''),
        '科室': patient_info.get('科室', ''),
        '住院号': patient_info.get('住院号', ''),
        '床号': patient_info.get('床号', ''),
        '处方号': patient_info.get('处方号', ''),
        '操作员ID': top_info.get('操作员ID', patient_info.get('操作员ID', extract_operator_id_from_text(result.get('full_text', '')))),
        '药品': result.get('medicines', []),
        '警告': result.get('warnings', []),
        '原始识别结果': result.get('full_text', '')
    }
    return structured


def process_image_path(
    img_path: str,
    recognizer: OCRRecognizer,
    task_prompt: str = '',
    fallback_name: Optional[str] = None
) -> tuple[Dict[str, Any], Optional[str]]:
    image = Image.open(img_path).convert('RGB')
    raw_image = image.copy()
    image = preprocess_image_for_ocr(image)

    patient_crop = crop_patient_block(image)
    patient_region_text = recognizer.recognize(patient_crop)
    patient_name_crop = crop_patient_name_block(image)
    patient_name_text = recognizer.recognize(patient_name_crop)
    result_text = recognizer.recognize(image, task_prompt)

    sections = split_prescription_sections(result_text)
    patient_section_text = "\n".join(sections['patient'])
    medicine_section_text = "\n".join(sections['medicine'])

    patient_info = extract_patient_info(patient_name_text)
    if not patient_info.get('姓名'):
        patient_info = extract_patient_info(patient_region_text)
    if not patient_info.get('姓名'):
        patient_info = extract_patient_info(patient_section_text)

    operator_id = recognize_top_right_number_with_tesseract(raw_image)
    if operator_id:
        patient_info['操作员ID'] = operator_id

    medicines = extract_medicine_info(medicine_section_text)
    patient_info, patient_warnings = check_and_fix_patient_info(
        patient_info,
        patient_region_text,
        result_text,
        fallback_name=fallback_name
    )
    medicines, medicine_warnings = check_and_fix_medicine_info(medicines, medicine_section_text, result_text)
    warnings = patient_warnings + medicine_warnings

    if patient_info.get('姓名') and re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', patient_info.get('姓名', '')):
        fallback_name = patient_info['姓名']

    image_result: Dict[str, Any] = {
        'full_text': result_text,
        'patient_region_text': patient_region_text,
        'patient_name_text': patient_name_text,
        'patient_section': sections['patient'],
        'medicine_section': sections['medicine'],
        'footer_section': sections['footer'],
        'patient_info': patient_info,
        'medicines': medicines,
        'warnings': warnings,
    }
    return image_result, fallback_name


def run_http_server(recognizer: OCRRecognizer, host: str = '127.0.0.1', port: int = 52100):
    class OCRRequestHandler(BaseHTTPRequestHandler):
        def _set_headers(self, code: int = 200, content_type: str = 'application/json') -> None:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.end_headers()

        def _send_json(self, data: Any, status_code: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self._set_headers(status_code)
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == '/status':
                self._send_json({'status': 'ok', 'message': 'OCR 服务已启动'})
            else:
                self._send_json({'error': '未找到接口'}, status_code=404)

        def do_POST(self) -> None:
            if self.path != '/ocr':
                self._send_json({'error': '未找到接口'}, status_code=404)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode('utf-8'))
            except Exception as exc:
                self._send_json({'error': f'JSON 解析失败: {exc}'}, status_code=400)
                return

            image_path = payload.get('image_path')
            image_base64 = payload.get('image_base64')
            task_prompt = payload.get('task_prompt', '')
            if not image_path and not image_base64:
                self._send_json({'error': '请提供 image_path 或 image_base64'}, status_code=400)
                return

            try:
                if image_path:
                    image = Image.open(image_path).convert('RGB')
                else:
                    image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
            except Exception as exc:
                self._send_json({'error': f'图片加载失败: {exc}'}, status_code=400)
                return

            try:
                image = preprocess_image_for_ocr(image)
                full_text = recognizer.recognize(image, task_prompt)

                sections = split_prescription_sections(full_text)
                patient_section_text = "\n".join(sections['patient'])
                medicine_section_text = "\n".join(sections['medicine'])

                patient_info = extract_patient_info(patient_section_text)
                if not patient_info.get('姓名'):
                    patient_info = extract_patient_info('\n'.join(sections['patient']))
                if not patient_info.get('姓名'):
                    patient_info = extract_patient_info(full_text)

                medicines = extract_medicine_info(medicine_section_text)
                patient_info, patient_warnings = check_and_fix_patient_info(patient_info, patient_section_text, full_text)
                medicines, medicine_warnings = check_and_fix_medicine_info(medicines, medicine_section_text, full_text)
                warnings = patient_warnings + medicine_warnings

                response = {
                    'full_text': full_text,
                    'patient_info': patient_info,
                    'medicines': medicines,
                    'warnings': warnings,
                    'sections': sections,
                }
                self._send_json(response)
            except Exception as exc:
                self._send_json({'error': f'OCR 处理失败: {exc}'}, status_code=500)

    server = ThreadingHTTPServer((host, port), OCRRequestHandler)
    print(f'HTTP OCR 服务已启动，访问地址 http://{host}:{port}/ocr')
    print('POST JSON: {"image_path":"..."} 或 {"image_base64":"..."}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nOCR 服务已停止。')
    finally:
        server.server_close()


# === 5. 程序入口与参数解析 ===
# 处理命令行参数、批量执行OCR、展示结果并写入文件

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_file = os.path.join(script_dir, 'server_results', 'ocr_results', 'ocr_results.json')

    parser = argparse.ArgumentParser(description='OCR识别服务')
    parser.add_argument('--input_dir', type=str, default=r'D:/测试', help='输入图片目录')
    parser.add_argument('--input_file', type=str, default='', help='单张图片输入路径')
    parser.add_argument('--output_file', type=str, default=default_output_file, help='输出文件路径')
    parser.add_argument('--task_prompt', type=str, default='', help='任务提示词')
    parser.add_argument('--fast_mode', action='store_true', help='仅使用 pytesseract 进行 OCR，跳过 GLM-OCR 模型，速度更快')
    parser.add_argument('--server', action='store_true', help='启动 HTTP 服务模式')
    parser.add_argument('--server_host', type=str, default='127.0.0.1', help='HTTP 服务监听地址')
    parser.add_argument('--server_port', type=int, default=52100, help='HTTP 服务监听端口')
    args = parser.parse_args()

    overall = tqdm(total=2, desc="全部流程", unit="步", bar_format='{desc}: {bar} {n}/{total} [{elapsed}<{remaining}]')

    if not args.fast_mode:
        # 检查并修复 transformers 依赖
        overall.set_description("检查依赖")
        ensure_transformers_installed()

    # 初始化OCR识别器
    overall.set_description("加载模型")
    recognizer = OCRRecognizer(fast_mode=args.fast_mode)
    overall.update(1)
    print(f"使用设备: {recognizer.DEVICE}")
    if args.fast_mode:
        print("    已启用快速模式：仅使用 pytesseract OCR，跳过模型推理。")

    if args.server:
        print(f"启动 HTTP 服务：{args.server_host}:{args.server_port}")
        run_http_server(recognizer, args.server_host, args.server_port)
        return

    if args.input_file:
        image_paths = [args.input_file]
        print(f"单张图片模式，输入文件：{args.input_file}")
    else:
        overall.set_description("查找图片")
        image_paths = find_images_in_directory(args.input_dir)
        overall.total = 3 + len(image_paths)
        overall.update(1)
        overall.refresh()
        print(f"\n找到 {len(image_paths)} 张图片\n")

    # 逐张识别
    all_results = {}
    last_patient_name: Optional[str] = None
    with tqdm(image_paths, desc="处理图片", unit="张", leave=False) as image_bar:
        for img_path in image_bar:
            print(f"\n--- {os.path.basename(img_path)} ---")
            try:
                image_result, last_patient_name = process_image_path(
                    img_path,
                    recognizer,
                    task_prompt=args.task_prompt,
                    fallback_name=last_patient_name
                )
                all_results[img_path] = image_result

                print(f"【完整识别结果】\n{image_result['full_text']}\n")
                print(f"【患者区识别结果】\n{image_result['patient_region_text']}\n")
                print(f"【患者姓名区识别结果】\n{image_result['patient_name_text']}\n")

                if image_result.get('warnings'):
                    print("【校验警告】")
                    for warn in image_result['warnings']:
                        print(f"  - {warn}")
                    print()

                patient_info = image_result.get('patient_info', {})
                if patient_info:
                    print("【患者信息】")
                    for key, value in patient_info.items():
                        print(f"  {key}: {value}")
                    print()

                medicines = image_result.get('medicines', [])
                if medicines:
                    print("【药品识别结果】")
                    print(f"药品数量: {len(medicines)}")
                    for j, med in enumerate(medicines, 1):
                        print(f"  药品 {j}:")
                        for key, value in med.items():
                            print(f"    {key}: {value}")
                        print()
                else:
                    print("【药品识别结果】\n未找到药品信息\n")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n错误: GPU显存不足，无法处理图片 {img_path}。程序将终止。")
                    try:
                        import torch
                        if hasattr(torch, 'cuda'):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    sys.exit(1)
                else:
                    print(f"处理图片 {img_path} 时发生错误: {e}")
                    all_results[img_path] = {'error': str(e)}
            except Exception as e:
                print(f"处理图片 {img_path} 时发生未知错误: {e}")
                all_results[img_path] = {'error': str(e)}

            overall.update(1)

    # 保存结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.output_file.lower().endswith('.txt'):
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(format_result_text(all_results))
    else:
        structured_results = []
        for img_path, result in all_results.items():
            structured_results.append(build_structured_result(img_path, result))
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_results, f, ensure_ascii=False, indent=2)

    overall.update(1)
    overall.close()
    print(f"\n识别结果已保存至: {args.output_file}")


if __name__ == "__main__":
    main()
