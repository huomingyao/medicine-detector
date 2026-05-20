"""
药品名称文本匹配模块
使用多种匹配策略识别OCR结果中的药品名称
"""
import re
from typing import Optional, List, Dict
from difflib import SequenceMatcher

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("建议安装fuzzywuzzy: pip install fuzzywuzzy")

try:
    import Levenshtein
    def levenshtein_distance(s1, s2):
        return Levenshtein.distance(s1, s2)
    def levenshtein_ratio(s1, s2):
        return Levenshtein.ratio(s1, s2)
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    def levenshtein_distance(s1, s2):
        """简单的Levenshtein距离实现"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def levenshtein_ratio(s1, s2):
        """计算相似度比率"""
        d = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (d / max_len)


class DrugTextMatcher:
    """药品名称匹配器"""

    # 药品名称中的关键词映射（用于快速匹配）
    KEYWORD_MAP = {
        '氨基己酸': '氨基己酸注射液',
        '氨甲环酸': '氨甲环酸注射液',
        '胞磷胆碱': '胞磷胆碱钠注射液',
        '单硝酸异山梨酯': '单硝酸异山梨酯注射液',
        '地塞米松': '地塞米松磷酸钠注射液',
        '多索茶碱': '多索茶碱注射液',
        '酚磺乙胺': '酚磺乙胺注射液',
        '氨基酸': '复方氨基酸注射液',
        '克林霉素': '克林霉素磷酸酯注射液',
        '硫酸镁': '硫酸镁注射液',
        '硫酸依替米星': '硫酸依替米星注射液',
        '硫辛酸': '硫辛酸注射液',
        '氯化钾': '氯化钾注射液',
        '氯化钠': '氯化钠注射液',
        '门冬氨酸钾镁': '门冬氨酸钾镁注射液',
        '浓氯化钠': '浓氯化钠注射液',
        '腺苷蛋氨酸': '注射用丁二磺酸腺苷蛋氨酸',
        '葡萄糖': '葡萄糖注射液',
        '维生素B6': '维生素B6注射液',
        '维生素C': '维生素C注射液',
        '西咪替丁': '西咪替丁注射液',
        '氨溴素': '盐酸氨溴素注射液',
        '法舒地尔': '盐酸法舒地尔注射液',
        '利多卡因': '盐酸利多卡因注射液',
        '异甘草酸镁': '异甘草酸镁注射液',
        '银杏': '银杏达莫注射液',
        '阿奇霉素': '注射用阿奇霉素',
        '比阿培南': '注射用比阿培南',
        '伏立康唑': '注射用伏立康唑',
        '美罗培南': '注射用美罗培南',
        '头孢呋辛': '注射用头孢呋辛钠',
    }

    def __init__(self, drug_list: List[str], threshold: int = 60):
        """
        初始化匹配器

        Args:
            drug_list: 药品名称列表
            threshold: 匹配阈值 (0-100)
        """
        self.drug_list = drug_list
        self.threshold = threshold
        self._build_drug_index()

    def _build_drug_index(self):
        """构建药品索引"""
        # 构建简洁的查找表
        self.drug_keywords = {}
        for drug in self.drug_list:
            # 提取关键词
            keywords = self._extract_keywords(drug)
            self.drug_keywords[drug] = keywords

    def _extract_keywords(self, drug_name: str) -> List[str]:
        """提取药品名称中的关键词"""
        keywords = []

        # 去除括号内容，提取核心名称
        core_name = re.sub(r'[/（(].*[)）]', '', drug_name)
        core_name = core_name.strip()

        # 提取中文关键词（2字以上）
        pattern = r'[一-龥]{2,}'
        matches = re.findall(pattern, core_name)
        keywords.extend(matches)

        return keywords

    def match(self, ocr_text: str, threshold: Optional[int] = None) -> List[dict]:
        """
        匹配OCR文字与药品列表

        Args:
            ocr_text: OCR识别的文字
            threshold: 匹配阈值（可选，覆盖默认阈值）

        Returns:
            匹配结果列表，按得分排序
            [
                {'drug': 'xxx', 'score': 85, 'method': 'fuzzy'},
                ...
            ]
        """
        if not ocr_text or not ocr_text.strip():
            return []

        threshold = threshold or self.threshold
        ocr_text = ocr_text.strip()
        results = []

        # 方法1: 精确包含匹配
        exact_matches = self._exact_match(ocr_text)
        results.extend(exact_matches)

        # 方法2: 子串匹配
        substring_matches = self._substring_match(ocr_text)
        for match in substring_matches:
            if not any(m['drug'] == match['drug'] for m in results):
                results.append(match)

        # 方法3: 模糊匹配
        fuzzy_matches = self._fuzzy_match(ocr_text)
        for match in fuzzy_matches:
            if not any(m['drug'] == match['drug'] for m in results):
                results.append(match)

        # 方法4: 关键词匹配
        keyword_matches = self._keyword_match(ocr_text)
        for match in keyword_matches:
            if not any(m['drug'] == match['drug'] for m in results):
                results.append(match)

        # 过滤低于阈值的結果
        results = [r for r in results if r['score'] >= threshold]

        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def _exact_match(self, ocr_text: str) -> List[dict]:
        """精确匹配（完全相等）"""
        results = []
        ocr_clean = self._clean_text(ocr_text)

        for drug in self.drug_list:
            drug_clean = self._clean_text(drug)
            if ocr_clean == drug_clean:
                results.append({
                    'drug': drug,
                    'score': 100,
                    'method': 'exact'
                })

        return results

    def _substring_match(self, ocr_text: str) -> List[dict]:
        """子串匹配（OCR文本包含药品名，或药品名包含OCR文本）"""
        results = []
        ocr_clean = self._clean_text(ocr_text)

        for drug in self.drug_list:
            drug_clean = self._clean_text(drug)

            # 检查是否相互包含
            if drug_clean in ocr_clean or ocr_clean in drug_clean:
                # 计算包含程度
                if len(drug_clean) > 0:
                    score = min(len(drug_clean), len(ocr_clean)) / max(len(drug_clean), len(ocr_clean)) * 100
                    results.append({
                        'drug': drug,
                        'score': int(score),
                        'method': 'substring'
                    })

        return results

    def _fuzzy_match(self, ocr_text: str, top_k: int = 5) -> List[dict]:
        """模糊匹配"""
        results = []
        ocr_clean = self._clean_text(ocr_text)

        if len(ocr_clean) < 2:
            return results

        # 计算与每个药品的相似度
        scores = []
        for drug in self.drug_list:
            drug_clean = self._clean_text(drug)

            # 多种相似度计算取最大值
            ratio = self._calc_similarity(ocr_clean, drug_clean)

            if ratio >= self.threshold:
                scores.append({
                    'drug': drug,
                    'score': int(ratio),
                    'method': 'fuzzy'
                })

        # 取top_k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def _keyword_match(self, ocr_text: str) -> List[dict]:
        """关键词匹配"""
        results = []
        ocr_clean = self._clean_text(ocr_text)

        for drug, keywords in self.drug_keywords.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword in ocr_clean:
                    matched_keywords.append(keyword)

            if matched_keywords:
                # 匹配率作为得分
                score = len(matched_keywords) / len(keywords) * 100
                # 基础分60 + 加权
                score = int(60 + score * 0.4)
                score = min(100, score)

                results.append({
                    'drug': drug,
                    'score': score,
                    'method': 'keyword',
                    'matched_keywords': matched_keywords
                })

        return results

    def _calc_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度"""
        if not s1 or not s2:
            return 0.0

        scores = []

        # 方法1: SequenceMatcher
        scores.append(SequenceMatcher(None, s1, s2).ratio() * 100)

        # 方法2: fuzzywuzzy (如果可用)
        if FUZZY_AVAILABLE:
            scores.append(fuzz.ratio(s1, s2))
            scores.append(fuzz.partial_ratio(s1, s2))
            scores.append(fuzz.token_sort_ratio(s1, s2))

        # 方法3: Levenshtein (如果可用)
        if LEVENSHTEIN_AVAILABLE:
            scores.append(levenshtein_ratio(s1, s2) * 100)

        # 返回最大值
        return max(scores) if scores else 0.0

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\w\s一-龥]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', '', text)
        return text.strip()

    def get_best_match(self, ocr_text: str) -> Optional[dict]:
        """
        获取最佳匹配结果

        Args:
            ocr_text: OCR识别的文字

        Returns:
            最佳匹配结果 {'drug': 'xxx', 'score': 85, 'method': 'fuzzy'}，无匹配时返回None
        """
        results = self.match(ocr_text)
        return results[0] if results else None


def load_drug_list_from_file(file_path: str) -> List[str]:
    """
    从文件加载药品列表

    Args:
        file_path: 药品列表文件路径（每行一个药品名）

    Returns:
        药品名称列表
    """
    drug_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    drug_list.append(line)
    except FileNotFoundError:
        print(f"药品列表文件不存在: {file_path}")
    except Exception as e:
        print(f"读取药品列表失败: {e}")

    return drug_list


