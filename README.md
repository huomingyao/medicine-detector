# 医院配液中心药品智能识别系统

本项目提供医院配液中心药品智能识别的多种技术方案。

## 三条技术路线

| 分支 | 技术方案 | 实现方式 |
|------|---------|---------|
| [main](tree/main) | 概览 | 总览和方案选择 |
| [yolo-direct](tree/yolo-direct) | 直接训练YOLO模型 | 直接训练YOLO识别药品类别 |
| [yolo+ocr](tree/yolo+ocr) | YOLO+OCR+LLM | YOLO检测→OCR识别→LLM推理 |
| [multimodal](tree/multimodal) | 向量库+多模态 | CLIP视觉匹配+向量检索 |

### 方案说明

**1. YOLO直接识别 (yolo-direct)**
- 直接训练YOLO模型识别药品类别
- 优点：速度快，纯本地无需API
- 缺点：需要大量标注数据
- 适用场景：药品包装统一、标准化的产品

**2. YOLO+OCR+LLM (yolo+ocr)**
- YOLO检测药瓶 → OCR识别文字 → LLM推理
- 优点：精度高，OCR+LLM双重校验
- 缺点：依赖OCR和LLM API
- 适用场景：文字清晰、需要高精度的产品

**3. 向量库+多模态 (multimodal)**
- CLIP视觉语义匹配 + 向量库检索
- 优点：无需训练，可快速上线新药品
- 缺点：依赖向量库
- 适用场景：药品种类多、需要快速迭代的产品

## 快速开始

```bash
# 克隆项目
git clone https://github.com/huomingyao/medicine-detector.git
cd medicine-detector

# 切换到指定方案分支
git checkout yolo+ocr     # YOLO+OCR+LLM方案
# 或
git checkout multimodal   # 向量库+多模态方案
# 或
git checkout yolo-direct   # YOLO直接识别方案
```

## 方案选择建议

| 场景 | 推荐方案 |
|------|---------|
| 药品包装统一、标准化的产品 | yolo-direct |
| 文字清晰、需要高精度 | yolo+ocr |
| 药品种类多、需要快速迭代 | multimodal |