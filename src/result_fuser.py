"""
YOLO与OCR结果融合模块
基于规则的结果融合决策
"""
from typing import Optional, Protocol, List
from dataclasses import dataclass
from enum import Enum


class DecisionStrategy(Enum):
    """决策策略"""
    OCR_PRIORITY = "ocr_priority"      # OCR优先
    YOLO_PRIORITY = "yolo_priority"    # YOLO优先
    FUSE_WEIGHTED = "fuse_weighted"    # 加权融合


@dataclass
class YOLOResult:
    """YOLO检测结果"""
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple = None  # (x1, y1, x2, y2)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    matched_drug: str
    match_score: int      # 0-100
    match_method: str    # 'exact', 'substring', 'fuzzy', 'keyword'
    raw_text: str = ""   # 原始OCR输出
    llm_raw_response: str = ""


@dataclass
class FusedResult:
    """融合结果"""
    final_drug: str
    confidence: float    # 0-1.0
    decision: str       # 'yolo', 'ocr', 'fused'
    source: str        # 来源说明
    yolo_conf: float = 0.0
    ocr_conf: float = 0.0
    yolo_class: str = ""
    ocr_drug: str = ""
    ocr_text: str = ""
    ocr_raw_text: str = ""
    llm_raw_response: str = ""
    bbox: tuple = None


class ResultFuser:
    """YOLO与OCR结果融合器"""

    # 融合阈值配置
    OCR_HIGH_THRESHOLD = 85   # OCR高分阈值
    OCR_MID_THRESHOLD = 60    # OCR中等阈值
    YOLO_HIGH_THRESHOLD = 0.9 # YOLO高置信度阈值
    YOLO_MID_THRESHOLD = 0.7   # YOLO中等置信度阈值

    def __init__(self,
                 ocr_high_threshold: int = 85,
                 ocr_mid_threshold: int = 60,
                 yolo_high_threshold: float = 0.9,
                 yolo_mid_threshold: float = 0.7):
        """
        初始化融合器

        Args:
            ocr_high_threshold: OCR高分阈值
            ocr_mid_threshold: OCR中等阈值
            yolo_high_threshold: YOLO高置信度阈值
            yolo_mid_threshold: YOLO中等置信度阈值
        """
        self.ocr_high_threshold = ocr_high_threshold
        self.ocr_mid_threshold = ocr_mid_threshold
        self.yolo_high_threshold = yolo_high_threshold
        self.yolo_mid_threshold = yolo_mid_threshold

    def fuse(self,
            yolo_result: Optional[YOLOResult],
            ocr_result: Optional[OCRResult]) -> FusedResult:
        """
        融合YOLO和OCR结果

        Args:
            yolo_result: YOLO检测结果
            ocr_result: OCR识别结果

        Returns:
            融合结果
        """
        # 处理边缘情况
        if yolo_result is None and ocr_result is None:
            return FusedResult(
                final_drug="未知",
                confidence=0.0,
                decision="none",
                source="YOLO和OCR均无结果"
            )

        if yolo_result is None:
            return self._ocr_only_decision(ocr_result)

        if ocr_result is None:
            return self._yolo_only_decision(yolo_result)

        # 两者都有结果，进行融合决策
        return self._fuse_decision(yolo_result, ocr_result)

    def _ocr_only_decision(self, ocr_result: OCRResult) -> FusedResult:
        """仅OCR结果决策"""
        if ocr_result.match_score >= self.ocr_high_threshold:
            return FusedResult(
                final_drug=ocr_result.matched_drug,
                confidence=ocr_result.match_score / 100.0,
                decision="ocr",
                source=f"OCR识别匹配成功 (得分: {ocr_result.match_score})",
                ocr_conf=ocr_result.match_score / 100.0,
                ocr_drug=ocr_result.matched_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', "")
            )
        elif ocr_result.match_score >= self.ocr_mid_threshold:
            return FusedResult(
                final_drug=ocr_result.matched_drug,
                confidence=ocr_result.match_score / 100.0 * 0.8,
                decision="ocr",
                source=f"OCR识别匹配中等 (得分: {ocr_result.match_score})",
                ocr_conf=ocr_result.match_score / 100.0,
                ocr_drug=ocr_result.matched_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', "")
            )
        else:
            return FusedResult(
                final_drug="未知",
                confidence=0.0,
                decision="none",
                source=f"OCR匹配失败 (得分: {ocr_result.match_score})",
                ocr_conf=ocr_result.match_score / 100.0,
                ocr_drug=ocr_result.matched_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', "")
            )

    def _yolo_only_decision(self, yolo_result: YOLOResult) -> FusedResult:
        """仅YOLO结果决策"""
        return FusedResult(
            final_drug=yolo_result.class_name,
            confidence=yolo_result.confidence,
            decision="yolo",
            source=f"YOLO检测 (置信度: {yolo_result.confidence:.2f})",
            yolo_conf=yolo_result.confidence,
            yolo_class=yolo_result.class_name,
            bbox=yolo_result.bbox
        )

    def _fuse_decision(self,
                       yolo_result: YOLOResult,
                       ocr_result: OCRResult) -> FusedResult:
        """融合决策 - OCR优先，YOLO只做定位"""
        ocr_score = ocr_result.match_score
        ocr_drug = ocr_result.matched_drug

        # YOLO只做定位，结果以OCR为准
        if ocr_score >= self.ocr_high_threshold:
            return FusedResult(
                final_drug=ocr_drug,
                confidence=ocr_score / 100.0,
                decision="ocr",
                source="OCR高分匹配",
                yolo_conf=yolo_result.confidence,
                ocr_conf=ocr_score / 100.0,
                yolo_class=yolo_result.class_name,
                ocr_drug=ocr_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', ""),
                bbox=yolo_result.bbox
            )
        elif ocr_score >= self.ocr_mid_threshold:
            return FusedResult(
                final_drug=ocr_drug,
                confidence=ocr_score / 100.0 * 0.8,
                decision="ocr",
                source="OCR中等匹配",
                yolo_conf=yolo_result.confidence,
                ocr_conf=ocr_score / 100.0,
                yolo_class=yolo_result.class_name,
                ocr_drug=ocr_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', ""),
                bbox=yolo_result.bbox
            )
        else:
            return FusedResult(
                final_drug="未知",
                confidence=0.0,
                decision="none",
                source="OCR无匹配",
                yolo_conf=yolo_result.confidence,
                ocr_conf=ocr_score / 100.0,
                yolo_class=yolo_result.class_name,
                ocr_drug=ocr_drug,
                ocr_text=ocr_result.text,
                ocr_raw_text=ocr_result.raw_text,
                llm_raw_response=getattr(ocr_result, 'llm_raw_response', ""),
                bbox=yolo_result.bbox
            )

    def get_multi_candidates(self,
                              yolo_results: List[YOLOResult],
                              ocr_results: List[OCRResult]) -> List[FusedResult]:
        """
        多候选结果融合

        Args:
            yolo_results: YOLO检测结果列表
            ocr_results: OCR识别结果列表

        Returns:
            融合后的候选列表
        """
        candidates = []

        # 标记已处理的YOLO结果
        yolo_used = set()

        # 先处理OCR结果
        for ocr in ocr_results:
            # 查找对应的YOLO结果
            matched_yolo = None
            for i, yolo in enumerate(yolo_results):
                if i not in yolo_used:
                    # 检查YOLO类别是否与OCR匹配
                    if yolo.class_name == ocr.matched_drug:
                        matched_yolo = yolo
                        yolo_used.add(i)
                        break

            # 融合
            if matched_yolo:
                fused = self.fuse(matched_yolo, ocr)
            else:
                fused = self._ocr_only_decision(ocr)
            candidates.append(fused)

        # 添加未匹配的YOLO结果
        for i, yolo in enumerate(yolo_results):
            if i not in yolo_used:
                fused = self._yolo_only_decision(yolo)
                candidates.append(fused)

        # 按置信度排序
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        return candidates


class WeightedFuser(ResultFuser):
    """加权融合器 - 使用加权投票融合"""

    def __init__(self, yolo_weight: float = 0.5, ocr_weight: float = 0.5, **kwargs):
        """
        初始化加权融合器

        Args:
            yolo_weight: YOLO权重
            ocr_weight: OCR权重
        """
        super().__init__(**kwargs)
        self.yolo_weight = yolo_weight
        self.ocr_weight = ocr_weight

    def _fuse_decision(self,
                       yolo_result: YOLOResult,
                       ocr_result: OCRResult) -> FusedResult:
        """加权融合决策"""
        yolo_conf = yolo_result.confidence
        ocr_score = ocr_result.match_score / 100.0

        # 加权计算置信度
        confidence = yolo_conf * self.yolo_weight + ocr_score * self.ocr_weight

        # 决定最终类别
        yolo_class = yolo_result.class_name
        ocr_drug = ocr_result.matched_drug

        if yolo_class == ocr_drug:
            final_drug = yolo_class
            decision = "fused"
            source = "YOLO和OCR结果一致"
        elif ocr_score > 0.6:
            final_drug = ocr_drug
            decision = "ocr"
            source = "OCR加权得分更高"
        else:
            final_drug = yolo_class
            decision = "yolo"
            source = "YOLO加权得分更高"

        return FusedResult(
            final_drug=final_drug,
            confidence=confidence,
            decision=decision,
            source=source,
            yolo_conf=yolo_conf,
            ocr_conf=ocr_score,
            yolo_class=yolo_class,
            ocr_drug=ocr_drug
        )


