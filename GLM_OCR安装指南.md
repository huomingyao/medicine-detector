# GLM-OCR 模型配置指南

本项目的 OCR 识别使用 GLM-OCR 模型，模型路径 `D:/GLM-OCR`。

## 环境要求

- Python 3.10+
- NVIDIA GPU (建议 8GB+ 显存，如 RTX 4070)
- CUDA 12.6
- 磁盘空间: 模型约 8-18GB

## 安装依赖

```bash
# 创建虚拟环境
conda create -n glm-ocr python=3.10
conda activate glm-ocr

# 安装 PyTorch CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install transformers accelerate pillow numpy tqdm
```

## 下载模型

模型下载到 `D:/GLM-OCR` 目录：

```bash
# 设置缓存路径
set HF_HOME=D:/GLM-OCR

# 使用 HuggingFace 镜像加速下载（可选）
set HF_ENDPOINT=https://hf-mirror.com

# Python 下载模型
python -c "from transformers import AutoModelForImageTextToText, AutoProcessor; \
model = AutoModelForImageTextToText.from_pretrained('THU-ML/GLM-OCR-v2', cache_dir='D:/GLM-OCR'); \
processor = AutoProcessor.from_pretrained('THU-ML/GLM-OCR-v2', cache_dir='D:/GLM-OCR')"
```

或从 HuggingFace 下载：https://huggingface.co/THU-ML/GLM-OCR-v2

## 显存优化

如果显存不足，可以使用 4-bit 量化：

```python
# 在 ocr_server.py 中启用
processor = AutoProcessor.from_pretrained(model_path, load_quantized=True, quantization_config=.bitsandbytes_config())
```

## 常见问题

### 1. CUDA 版本不匹配
```bash
python -c "import torch; print(torch.version.cuda)"
```

### 2. 模型下载慢
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### 3. 显存不足 (OOM)
- 启用 4-bit 量化
- 减小图片尺寸
- 使用更小的模型

### 4. 模型路径错误
确保模型在 `D:/GLM-OCR` 目录，或在代码中修改路径。

## 模型选择

| 模型 | 显存要求 |
|------|----------|
| THU-ML/GLM-OCR-v2 | ~8GB |
| openbmb/MiniCPM-V-2_6 | ~10GB |
| Qwen/Qwen2-VL-7B-Instruct | ~16GB |