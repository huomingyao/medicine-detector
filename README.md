# 医院配液中心药品智能识别系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/YOLO-v8-orange" alt="YOLO">
  <img src="https://img.shields.io/badge/OCR-Tesseract-green" alt="OCR">
</p>

## 应用背景

本系统专为**医院配液中心**设计，用于自动识别药瓶上的药品信息，帮助药师快速准确地核对药品，避免配药错误。

### 解决的问题

1. **工作效率低**: 传统人工核对药瓶标签费时费力
2. **人为错误**: 药师在高强度工作下容易看错药品
3. **信息核对难**: 药瓶标签字体小、不清晰，难以快速辨认

### 应用场景

- 药房配药自动核对
- 药品入库扫码识别
- 库存药品盘点和查询
- 药品分发自动化流水线

### 技术方案

系统采用**多模态融合识别**技术，结合三种识别方式互相印证：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   YOLO      │     │    OCR      │     │    LLM      │
│  目标检测   │     │  文字识别   │     │   语义推理  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                            │
                     ┌──────▼──────┐
                     │  结果融合   │
                     │  决策引擎   │
                     └─────────────┘
```

**识别方式说明**：

| 识别方式 | 技术 | 优点 | 适用场景 |
|---------|------|------|----------|
| **YOLO直接识别** | 目标检测模型 | 速度快，可同时识别多个药品 | 常见药品包装识别 |
| **OCR文字识别** | Tesseract | 识别药瓶上的文字信息 | 文字清晰的药瓶 |
| **LLM语义推理** | 大语言模型 | 可推理模糊、不完整信息 | 文字模糊或缺失时 |

### 识别流程

1. **YOLO检测**: 使用YOLOv8模型检测药瓶区域，识别药品类别
2. **OCR识别**: 对检测到的药瓶区域进行文字识别
3. **文本匹配**: 将识别文字与药品库进行匹配
4. **LLM推理**: 使用大语言模型进行语义推理（可选）
5. **结果融合**: 综合三种结果，输出最终识别结果

## 文件说明

### 核心模块 (src/)

| 文件 | 功能 |
|------|------|
| [src/drug_recognizer.py](src/drug_recognizer.py) | 主识别器，整合YOLO+OCR+LLM进行药品识别 |
| [src/ocr_server.py](src/ocr_server.py) | OCR文字识别 (pytesseract) |
| [src/text_matcher.py](src/text_matcher.py) | 药品名称文本匹配 |
| [src/result_fuser.py](src/result_fuser.py) | YOLO与OCR结果融合决策 |

### 脚本

| 文件 | 功能 |
|------|------|
| [inference.py](inference.py) | 命令行推理工具 |
| [build_drug_library.py](build_drug_library.py) | 构建药瓶文字库 |

### 数据文件

| 文件 | 功能 |
|------|------|
| [classes.txt](classes.txt) | 药品名称列表 |
| [drug_library.json](drug_library.json) | 药瓶文字库 (OCR参考数据) |
| [yolov8x-v8.2.0.pt](yolov8x-v8.2.0.pt) | YOLO模型权重 |

## 环境配置

```bash
# 安装依赖
pip install ultralytics pytesseract pillow numpy openai

# 配置 minimax API 密钥
set MINIMAX_API=your_api_key_here
# 兼容旧变量名
set LLM_API_KEY=your_api_key_here
set GLM_API_KEY=your_api_key_here
```

需要安装 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) 并添加到系统路径。

## 使用方法

### 方法1: 命令行推理

```bash
# 药品列表模式
python inference.py --yolo yolov8x-v8.2.0.pt --image test/img.jpg

# 文字库模式 (使用LLM推理)
python inference.py --yolo yolov8x-v8.2.0.pt --image test/img.jpg --library drug_library.json

# 批量识别
python inference.py --yolo yolov8x-v8.2.0.pt --input test/ --output results.json
```

### 方法2: Python API

```python
import sys
sys.path.insert(0, 'src')
from drug_recognizer import DrugRecognizer

# 初始化识别器
recognizer = DrugRecognizer(
    yolo_model_path='yolov8x-v8.2.0.pt',
    library_path='drug_library.json',  # 使用文字库
    use_llm=True,           # 启用LLM推理
    llm_model='minimax'
)

# 识别单张图片
result = recognizer.recognize('test/img.jpg', use_ocr=True)

# 获取结果
for fused in result.fused_results:
    print(f"药品: {fused.final_drug}, 置信度: {fused.confidence:.2f}")
```

## 参数说明

### DrugRecognizer 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| yolo_model_path | str | 必填 | YOLO模型路径 |
| drug_list | List[str] | None | 药品列表 (与classes.txt配合) |
| library_path | str | None | 文字库路径 (JSON) |
| conf_threshold | float | 0.25 | YOLO置信度阈值 |
| iou_threshold | float | 0.45 | YOLO IOU阈值 |
| fast_mode | bool | True | True=pytesseract, False=GLM模型 |
| use_llm | bool | True | 使用 minimax LLM 推理 |
| llm_model | str | minimax | LLM模型名称 |

### inference.py 参数

| 参数 | 简写 | 说明 |
|------|------|------|
| --yolo | -y | YOLO模型路径 (必填) |
| --image | -i | 单张图片路径 |
| --input | -I | 图片目录 |
| --library | -l | 文字库文件 |
| --output | -o | 输出结果文件 |
| --no-ocr | | 不使用OCR |
| --no-llm | | 不使用LLM推理 |
| --verbose | -v | 详细输出 |

## 输出格式

```json
{
  "image_path": "test/img.jpg",
  "detection_count": 2,
  "detections": [
    {
      "drug": "注射用阿奇霉素",
      "confidence": 0.95,
      "decision": "ocr",
      "source": "YOLO和OCR结果一致，互相印证",
      "yolo_conf": 0.92,
      "ocr_conf": 0.90
    }
  ],
  "time": {
    "yolo": 0.50,
    "ocr": 1.20,
    "llm": 0.80,
    "total": 2.50
  }
}
```

## 决策逻辑

系统采用多级决策策略，综合考虑YOLO、OCR和LLM三种识别结果：

- **YOLO高 + OCR高 + 一致**: 加权提升置信度，结果互相印证
- **YOLO高 + OCR低**: 采用YOLO结果，YOLO识别更可靠
- **YOLO低 + OCR高**: 采用OCR结果，文字识别更可靠
- **仅YOLO**: 使用YOLO直接识别结果
- **仅OCR**: 使用OCR文本匹配结果
- **YOLO+LLM**: 当OCR无法识别时，使用LLM推理药品信息

### YOLO直接识别模式

当使用YOLO直接识别模式时，系统通过YOLOv8目标检测模型直接输出药品类别：

```python
# 使用YOLO直接识别（不依赖OCR）
recognizer = DrugRecognizer(
    yolo_model_path='yolov8x-v8.2.0.pt',
    use_llm=False  # 不使用LLM
)

# 识别图片 - YOLO直接输出药品类别
result = recognizer.recognize('test/img.jpg', use_ocr=False)
```

**YOLO直接识别的优势**：
- 速度快（无需OCR处理）
- 可批量处理多个药瓶
- 适用于标准药品包装识别

**适用场景**：
- 药品包装规范、清晰
- 需要快速批量识别
- 光照条件良好的环境