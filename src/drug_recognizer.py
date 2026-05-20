"""
药品识别器 - YOLO + OCR + LLM 推理
医院配液中心药瓶识别系统

识别流程:
1. YOLO 检测药瓶位置
2. OCR (GLM-OCR/pytesseract) 识别药瓶文字
3. 将 OCR 结果 + 文字库 发送给 LLM API 进行推理判断（阿里云百炼 deepseek-r1）

环境变量:
  DASHSCOPE_API_KEY: 阿里云百炼 API 密钥（deepr1-seek-r1）
  兼容旧变量: MINIMAX_API / LLM_API_KEY / GLM_API_KEY
  LLM_MODEL: LLM模型名称 (默认: deepseek-r1)
"""
import os
import time
import logging
import json
import re
from pathlib import Path
from typing import Optional, Union, List
from dataclasses import dataclass
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入本地OCR模块
from ocr_server import OCRRecognizer

# 大模型调用配置（阿里云百炼 dashscope）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MINIMAX_API") or os.environ.get("LLM_API_KEY") or os.environ.get("GLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")  # 主模型
LLM_BACKUP_MODEL = os.environ.get("LLM_BACKUP_MODEL", "qwen-turbo")  # 备用模型
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE = 1.0
LLM_MIN_INTERVAL = 2.0
_last_llm_call_time = 0.0

# 导入YOLO
try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics未安装，请运行: pip install ultralytics")
    YOLO = None

# 导入本地模块
from text_matcher import DrugTextMatcher
from result_fuser import ResultFuser, YOLOResult, OCRResult, FusedResult


@dataclass
class DetectionResult:
    """检测结果"""
    image_path: str
    detections: list  # YOLO检测结果
    fused_results: list  # 融合后的结果
    total_time: float  # 总耗时
    yolo_time: float  # YOLO耗时
    ocr_time: float  # OCR耗时
    llm_time: float  # LLM推理耗时


# ==================== LLM 推理相关 ====================

def call_glm_llm_api(api_key: str = None, messages: list = None, model: str = None, temperature: float = 0.3, max_retries: int = LLM_MAX_RETRIES) -> str:
    """
    调用阿里云百炼 LLM API，带备用模型支持

    Args:
        api_key: API密钥
        messages: 消息列表 [{"role": "user", "content": "..."}]
        model: 模型名称 (默认: deepseek-r1)
        temperature: 温度参数
        max_retries: 最大重试次数

    Returns:
        模型回复
    """
    global _last_llm_call_time

    api_key = api_key or DASHSCOPE_API_KEY
    model = model or LLM_MODEL
    backup_model = LLM_BACKUP_MODEL

    if not api_key:
        return "调用失败: 未提供 DASHSCOPE_API_KEY 环境变量"

    now = time.time()
    elapsed = now - _last_llm_call_time
    if elapsed < LLM_MIN_INTERVAL:
        wait_time = LLM_MIN_INTERVAL - elapsed
        logger.info(f"LLM 请求过于频繁，等待 {wait_time:.2f}s 后再发起")
        time.sleep(wait_time)

    import dashscope
    dashscope.api_key = api_key

    # 需要切换模型 的错误码
    token_error_codes = ['token_limit_exceeded', 'insufficient_quota', 'billing_exceeded', 'max_tokens exceeded']

    def _call_with_model(model_name: str, messages: list, temperature: float) -> tuple:
        """尝试调用指定模型"""
        try:
            resp = dashscope.Generation.call(
                model=model_name,
                messages=messages,
                temperature=temperature,
                result_format='message'
            )
            _last_llm_call_time = time.time()
            return resp, None
        except Exception as e:
            return None, str(e)

    # 尝试主模型
    for attempt in range(1, max_retries + 1):
        resp, err = _call_with_model(model, messages, temperature)

        if resp and resp.status_code == 200:
            return resp.output.choices[0].message.content
        elif resp:
            err_msg = resp.message or ""
            err_code = resp.code or ""
            # 检查是否需要切换到备用模型
            need_backup = any(code in str(err_msg).lower() or code in str(err_code).lower() for code in token_error_codes)
            if need_backup:
                logger.warning(f"主模型 {model} token不足，切换到备用模型 {backup_model}")
                break  # 跳出主模型重试循环，去尝试备用模型
            else:
                logger.error(f"LLM调用失败: {err_code} - {err_msg}")
                return f"调用失败: {err_msg}"
        else:
            logger.error(f"LLM调用失败: {err}")

        backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
        logger.warning(f"主模型{model}调用失败，{attempt}/{max_retries}，等待 {backoff:.1f}s 重试")
        time.sleep(backoff)

    # 备用模型重试
    logger.info(f"使用备用模型 {backup_model} 进行推理")
    for attempt in range(1, max_retries + 1):
        resp, err = _call_with_model(backup_model, messages, temperature)

        if resp and resp.status_code == 200:
            logger.info(f"备用模型 {backup_model} 调用成功")
            return resp.output.choices[0].message.content
        elif resp:
            logger.error(f"备用模型调用失败: {resp.code} - {resp.message}")
            return f"调用失败: {resp.message}"
        else:
            logger.error(f"备用模型调用失败: {err}")

        backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
        logger.warning(f"��用模型调用失败，{attempt}/{max_retries}，等待 {backoff:.1f}s 重试")
        time.sleep(backoff)

    return "调用失败: 所有模型均不可用"


def build_llm_prompt(ocr_text: str, library: dict) -> list:
    """
    构建LLM推理的提示词

    Args:
        ocr_text: OCR识别到的文字
        library: 文字库内容

    Returns:
        消息列表
    """
    # 构建文字库信息
    library_info = []
    for drug_name, drug_data in library.items():
        info = f"药瓶: {drug_name}\n"
        info += f"  药品名称: {drug_data.get('药品名称', '')}\n"
        info += f"  规格: {drug_data.get('规格', '')}\n"
        info += f"  剂型: {drug_data.get('剂型', '')}\n"
        info += f"  厂家: {drug_data.get('厂家', '')}\n"
        info += f"  完整描述: {drug_data.get('all_text', '')}\n"
        library_info.append(info)

    library_text = "\n".join(library_info)

    prompt = f"""你是一个药品识别助手。请根据OCR识别到的药瓶文字，从文字库中找出最匹配的药品。

## OCR识别结果（可能不完整或不准确）:
{ocr_text}

## 文字库（已知的药瓶信息）:
{library_text}

## 任务:
1. 根据OCR识别结果，从文字库中找出最匹配的药品
2. 考虑OCR可能存在识别错误的情况，进行模糊匹配
3. 输出匹配到的药品名称和匹配理由

## 输出格式:
匹配结果: [药品名称]
匹配理由: [简短说明]
置信度: [高/中/低]
"""

    return [
        {"role": "user", "content": prompt}
    ]


def parse_llm_response(response: str) -> dict:
    """解析LLM回复"""
    result = {
        'matched_drug': '',
        'reason': '',
        'confidence': '中'
    }

    # 提取匹配结果
    match = re.search(r'匹配结果[:：]\s*(.+?)(?:\n|$)', response)
    if match:
        result['matched_drug'] = match.group(1).strip()

    # 提取匹配理由
    match = re.search(r'匹配理由[:：]\s*(.+?)(?:\n|$)', response)
    if match:
        result['reason'] = match.group(1).strip()

    # 提取置信度
    match = re.search(r'置信度[:：]\s*(高|中|低)', response)
    if match:
        result['confidence'] = match.group(1).strip()

    return result


# ==================== 识别器主类 ====================

class DrugRecognizer:
    """药品识别器 - YOLO + OCR + 阿里云百炼 LLM 推理"""

    def __init__(self,
                 yolo_model_path: str,
                 drug_list: List[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 fast_mode: bool = True,
                 library_path: str = None,
                 use_llm: bool = True,
                 llm_model: str = None,
                 target_classes: List[str] = ["bottle"]):
        """
        初始化识别器

        Args:
            yolo_model_path: YOLO模型路径
            drug_list: 药品名称列表（用于匹配）
            conf_threshold: YOLO置信度阈值
            iou_threshold: YOLO IOU阈值
            fast_mode: 是否使用快速模式 (True=pytesseract, False=GLM本地模型)
            library_path: 药瓶文字库文件路径
            use_llm: 是否使用LLM推理（默认True）
            llm_model: LLM模型名称
            target_classes: YOLO目标类别列表（只对这些类别进行OCR识别，如 ["bottle", "bowl"]）
        """
        self.yolo_model_path = yolo_model_path
        self.drug_list = drug_list or []
        self.library_path = library_path
        self.library = None
        self.use_library = False
        self.use_llm = use_llm

        # 目标类别过滤列表
        self.target_classes = set(target_classes) if target_classes else None

        # 获取API密钥，优先使用 DASHSCOPE_API_KEY（阿里云百炼），兼容旧变量
        self.api_key = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('MINIMAX_API') or os.environ.get('LLM_API_KEY') or os.environ.get('GLM_API_KEY', '')
        api_key_source = 'DASHSCOPE_API_KEY' if os.environ.get('DASHSCOPE_API_KEY') else ('MINIMAX_API' if os.environ.get('MINIMAX_API') else ('LLM_API_KEY' if os.environ.get('LLM_API_KEY') else ('GLM_API_KEY' if os.environ.get('GLM_API_KEY') else '')))
        if not self.api_key and use_llm:
            logger.warning("未设置 DASHSCOPE_API_KEY、MINIMAX_API、LLM_API_KEY 或 GLM_API_KEY 环境变量，LLM推理将不可用")

        # 模型选择，默认使用 deepseek-r1
        if llm_model:
            self.llm_model = llm_model
        elif os.environ.get('LLM_MODEL'):
            self.llm_model = os.environ.get('LLM_MODEL')
        else:
            self.llm_model = LLM_MODEL

        logger.info(f"LLM API Key 来源: {api_key_source or 'none'}, 使用模型: {self.llm_model}")

        # 加载YOLO模型
        logger.info(f"加载YOLO模型: {yolo_model_path}")
        if YOLO is None:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        self.yolo = YOLO(yolo_model_path)
        # 强制使用 CPU
        self.yolo.to('cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 初始化本地OCR
        logger.info(f"初始化本地OCR (fast_mode={fast_mode})")
        self.ocr = OCRRecognizer(fast_mode=fast_mode)

        # 加载文字库
        if library_path and os.path.exists(library_path):
            logger.info(f"加载药瓶文字库: {library_path}")
            self._load_library(library_path)

        # 初始化文本匹配器
        logger.info("初始化文本匹配器")
        if self.use_library:
            library_drugs = list(self.library.keys()) if self.library else []
            self.matcher = DrugTextMatcher(library_drugs, threshold=50)
        else:
            self.matcher = DrugTextMatcher(self.drug_list, threshold=60)

        # 初始化结果融合器
        logger.info("初始化结果融合器")
        self.fuser = ResultFuser()

        logger.info("药品识别器初始化完成")

    def _load_library(self, library_path: str):
        """加载药瓶文字库"""
        try:
            with open(library_path, 'r', encoding='utf-8') as f:
                self.library = json.load(f)
            self.use_library = True
            logger.info(f"文字库加载成功: {len(self.library)} 个药瓶")
        except Exception as e:
            logger.error(f"加载文字库失败: {e}")
            self.library = None

    def _match_with_library(self, ocr_text: str) -> Optional[dict]:
        """使用文字库匹配（纯文本匹配，不用LLM）"""
        if not self.library or not ocr_text:
            return None

        from difflib import SequenceMatcher

        best_match = None
        best_score = 0

        for drug_name, drug_data in self.library.items():
            all_text = drug_data.get('all_text', '')
            if not all_text:
                continue

            # 相似度计算
            text1 = ocr_text[:100] if len(ocr_text) > 100 else ocr_text
            text2 = all_text[:100] if len(all_text) > 100 else all_text

            ratio = SequenceMatcher(None, text1, text2).ratio() * 100

            if ratio > best_score:
                best_score = ratio
                best_match = {
                    'drug': drug_name,
                    'score': int(ratio),
                    'method': 'library_fuzzy'
                }

        if best_match and best_score < 50:
            logger.info(f"文字库最佳候选（低分）: {best_match['drug']} (score={best_score:.1f})")

        return best_match

    def _match_with_llm(self, ocr_text: str) -> Optional[dict]:
        """使用阿里云百炼 LLM 进行推理匹配"""
        if not self.api_key or not self.library:
            return None

        try:
            # 构建提示词
            messages = build_llm_prompt(ocr_text, self.library)

            # 调用API
            response = call_glm_llm_api(self.api_key, messages, model=self.llm_model)
            if isinstance(response, str) and response.startswith("调用失败:"):
                logger.warning(f"LLM调用失败: {response}")
                return None

            # 解析结果
            result = parse_llm_response(response)

            if result['matched_drug']:
                # 映射置信度到分数
                confidence_map = {'高': 90, '中': 70, '低': 50}
                score = confidence_map.get(result['confidence'], 60)

                return {
                    'drug': result['matched_drug'],
                    'score': score,
                    'method': 'llm',
                    'reason': result['reason'],
                    'raw_response': response
                }

        except Exception as e:
            logger.error(f"LLM推理失败: {e}")

        return None

    def recognize(self,
                image_path: str,
                use_ocr: bool = True,
                crop_margin: int = 10) -> DetectionResult:
        """
        识别图片中的药品

        Args:
            image_path: 图片路径
            use_ocr: 是否使用OCR
            crop_margin: 裁剪边距

        Returns:
            DetectionResult
        """
        start_time = time.time()

        # YOLO检测
        yolo_start = time.time()
        yolo_results = self._yolo_detect(image_path)
        yolo_time = time.time() - yolo_start

        logger.info(f"YOLO检测完成，发现 {len(yolo_results)} 个目标，耗时 {yolo_time:.2f}s")

        # OCR识别和融合
        ocr_time = 0.0
        llm_time = 0.0
        fused_results = []

        if use_ocr and yolo_results:
            ocr_start = time.time()

            for yolo_res in yolo_results:
                # 裁剪药瓶区域
                crop = self._crop_region(image_path, yolo_res.bbox, crop_margin)

                # OCR识别文字
                ocr_result = self._ocr_recognize(crop)
                logger.info(f"OCR识别文本: {ocr_result['text']!r}")

                # 文本匹配
                if ocr_result['success'] and ocr_result['text']:
                    match_result = None

                    # 根据模式选择匹配方法
                    if self.use_library and self.use_llm:
                        # 优先使用LLM推理
                        llm_start = time.time()
                        match_result = self._match_with_llm(ocr_result['text'])
                        llm_time += time.time() - llm_start

                        if not match_result:
                            # LLM失败时回退到文本匹配
                            match_result = self._match_with_library(ocr_result['text'])
                    elif self.use_library:
                        match_result = self._match_with_library(ocr_result['text'])
                    else:
                        match_result = self.matcher.get_best_match(ocr_result['text'])

                    if match_result:
                        logger.info(f"匹配结果: {match_result.get('drug')} (score={match_result.get('score')}, method={match_result.get('method')})")
                        ocr_data = OCRResult(
                            text=ocr_result['text'],
                            matched_drug=match_result['drug'],
                            match_score=match_result['score'],
                            match_method=match_result['method'],
                            raw_text=ocr_result.get('raw', ''),
                            llm_raw_response=match_result.get('raw_response', '')
                        )
                    else:
                        logger.warning(f"OCR文本未匹配: {ocr_result['text']!r}")
                        if not self.use_library:
                            candidates = self.matcher.match(ocr_result['text'], threshold=0)
                            if candidates:
                                logger.info(f"候选匹配列表: {candidates[:5]}")
                        ocr_data = OCRResult(
                            text=ocr_result['text'],
                            matched_drug="未知",
                            match_score=0,
                            match_method="none",
                            raw_text=ocr_result.get('raw', ''),
                            llm_raw_response=match_result.get('raw_response', '') if match_result else ""
                        )
                else:
                    if ocr_result['text']:
                        logger.warning(f"OCR识别到的文本无效，未进入匹配: {ocr_result['text']!r}")
                    else:
                        logger.warning("OCR未识别到文字")
                    ocr_data = None

                # 融合结果
                fused = self.fuser.fuse(yolo_res, ocr_data)
                fused_results.append(fused)

            ocr_time = time.time() - ocr_start
            logger.info(f"OCR识别完成，耗时 {ocr_time:.2f}s")

            if llm_time > 0:
                logger.info(f"LLM推理完成，耗时 {llm_time:.2f}s")

        else:
            # 仅使用YOLO结果
            for yolo_res in yolo_results:
                fused = self.fuser.fuse(yolo_res, None)
                fused_results.append(fused)

        total_time = time.time() - start_time

        return DetectionResult(
            image_path=image_path,
            detections=yolo_results,
            fused_results=fused_results,
            total_time=total_time,
            yolo_time=yolo_time,
            ocr_time=ocr_time,
            llm_time=llm_time
        )

    def recognize_batch(self,
                    image_paths: List[str],
                    use_ocr: bool = True) -> List[DetectionResult]:
        """批量识别"""
        results = []
        for img_path in image_paths:
            logger.info(f"识别图片: {img_path}")
            result = self.recognize(img_path, use_ocr=use_ocr)
            results.append(result)
        return results

    def _yolo_detect(self, image_path: str) -> list:
        """YOLO目标检测"""
        try:
            results = self.yolo(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            yolo_results = []
            result = results[0]

            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    class_name = result.names[class_id] if result.names else f"class_{class_id}"

                    # 过滤目标类别
                    if self.target_classes and class_name.lower() not in self.target_classes:
                        logger.info(f"跳过非目标类别: {class_name}")
                        continue

                    box = boxes.xyxy[i].cpu().numpy()
                    bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

                    yolo_results.append(YOLOResult(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        bbox=bbox
                    ))

            return yolo_results

        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return []

    def _ocr_recognize(self, image_crop: np.ndarray) -> dict:
        """OCR识别"""
        try:
            from PIL import Image

            if isinstance(image_crop, np.ndarray):
                pil_image = Image.fromarray(image_crop)
            else:
                pil_image = image_crop

            text = self.ocr.recognize(pil_image)

            return {
                'success': bool(text),
                'text': text,
                'raw': text
            }

        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return {'success': False, 'text': '', 'error': str(e)}

    def _crop_region(self, image_path: str, bbox: tuple, margin: int = 10) -> np.ndarray:
        """裁剪图片区域"""
        from PIL import Image

        pil_img = Image.open(image_path).convert('RGB')
        w, h = pil_img.size

        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        crop = pil_img.crop((x1, y1, x2, y2))
        return np.array(crop)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'yolo_model': self.yolo_model_path,
            'drug_count': len(self.drug_list),
            'use_library': self.use_library,
            'library_count': len(self.library) if self.library else 0,
            'use_llm': self.use_llm and bool(self.api_key),
            'llm_model': self.llm_model
        }

