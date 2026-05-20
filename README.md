# 医院配液中心药品智能识别系统

基于 YOLO + OCR + 阿里云百炼 LLM 的药瓶识别系统。

## 系统功能

本系统用于识别图片中的药瓶，并识别药瓶上的药品名称。

### 识别流程

```
输入图片 → YOLO检测药瓶 → 裁剪药瓶区域 → OCR识别文字 → 文本匹配/LLM推理 → 输出药品名称
```

1. **YOLO检测**: 使用YOLOv8检测图片中的药瓶(bottle)位置
2. **OCR识别**: 对检测到的药瓶区域进行文字识别
3. **文本匹配**: 将识别文字与药品库进行匹配
4. **LLM推理**: 使用阿里云百炼LLM进行语义推理（可选）
5. **结果融合**: 综合YOLO和OCR结果，输出最终识别结果

### 技术特点

- **YOLO**: 仅检测"bottle"（药瓶）目标，不直接输出药品名称
- **OCR**: 使用GLM-OCR进行文字识别，默认从 `D:/GLM-OCR` 加载模型
- **LLM**: 阿里云百炼dashscope API，支持deepseek-r1、qwen-plus等模型
- **结果融合**: 综合目标检测和文字识别结果，智能输出最终药品名称

## 文件说明

```
├── src/
│   ├── drug_recognizer.py    # 主识别器（YOLO+OCR+LLM）
│   ├── ocr_server.py      # OCR文字识别
│   ├── text_matcher.py  # 文本匹配
│   └── result_fuser.py  # 结果融合
├── inference.py          # 命令行推理工具
├── build_drug_library.py # 构建药品文字库
├── classes.txt         # 药品列表
├── drug_library.json   # 药品文字库
└── yolov8x-v8.2.0.pt # YOLO模型
```

## 环境配置

```bash
# Python 版本要求
Python >= 3.8

# 安装依赖
pip install ultralytics>=8.0.0 pillow>=10.0.0 dashscope>=1.14.0 fuzzywuzzy>=0.18.0 python-Levenshtein>=0.12.0 numpy>=1.20.0 tqdm>=4.60.0 transformers>=4.30.0 torch>=2.0.0

# 需要下载 GLM-OCR 模型并放到 D:/GLM-OCR 目录
# 模型地址: https://huggingface.co/THU-ML/GLM-OCR

# 配置阿里云百炼API密钥
set DASHSCOPE_API_KEY=your_api_key_here

# 可选：配置LLM模型
set LLM_MODEL=deepseek-r1
```

## 使用方法

### 命令行推理

编辑 `inference.py` 中的配置区域：

```python
YOLO_MODEL_PATH = "yolov8x-v8.2.0.pt"
IMAGE_PATH = "test.jpg"  # 或图片目录
USE_OCR = True
USE_LLM = True
```

运行：
```bash
python inference.py
```

### Python API

```python
from src.drug_recognizer import DrugRecognizer

recognizer = DrugRecognizer(
    yolo_model_path='yolov8x-v8.2.0.pt',
    library_path='drug_library.json',
    use_llm=True
)

result = recognizer.recognize('test.jpg', use_ocr=True)

for fused in result.fused_results:
    print(f"药品: {fused.final_drug}")
```

## 输出示例

```json
{
  "image_path": "test.jpg",
  "detection_count": 2,
  "detections": [
    {
      "drug": "注射用阿奇霉素",
      "confidence": 0.85,
      "decision": "yolo+ocr",
      "source": "YOLO检测到药瓶，OCR识别文字匹配",
      "yolo_conf": 0.92,
      "ocr_conf": 0.78
    }
  ],
  "time": {
    "yolo": 0.15,
    "ocr": 1.20,
    "llm": 0.80,
    "total": 2.15
  }
}
```

## 调用LLM说明

当 OCR 识别文字模糊或不准确时，系统会调用阿里云百炼的 LLM 进行语义推理：

```python
# 构建提示词
messages = [
    {"role": "user", "content": "请根据OCR识别到的文字，从文字库中找出匹配的药品..."}
]

# 调用API
from dashscope import Generation
response = Generation.call(model='deepseek-r1', messages=messages)
```

API密钥获取：https://dashscope.console.aliyun.com/