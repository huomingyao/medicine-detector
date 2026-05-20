# GLM-OCR Windows 部署指南 (CUDA 12.6)

基于智谱 AI GLM-4 的 OCR 识别服务，支持 Windows 系统 + CUDA 12.6 部署。

## 文件说明

| 文件 | 说明 |
|------|------|
| `ocr_server.py` | OCR 服务器 (FastAPI) |
| `ocr_client.py` | Python 客户端调用示例 |
| `ocr_local.py` | 本地直接使用脚本 |
| `install.bat` | 一键安装依赖脚本 |
| `start_server.bat` | 一键启动服务器脚本 |

## 快速开始

### 1. 环境要求

- Windows 10/11
- Python 3.10+
- NVIDIA GPU (建议 8GB+ 显存，如 RTX 4070)
- CUDA 12.6
- 磁盘空间: 模型约 8-18GB + 缓存空间

### 2. 安装依赖

```bash
# 创建虚拟环境
conda create -n glm-ocr python=3.10
conda activate glm-ocr

# 安装 PyTorch CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install transformers accelerate pillow fastapi uvicorn python-multipart requests tiktoken

# 如需 4-bit 量化支持（8GB 显存推荐）
pip install bitsandbytes
```

### 3. 启动服务器

```bash
# 方式1：双击运行
start_server.bat

# 方式2：命令行
venv\Scripts\activate
python ocr_server.py
```

服务器启动后访问：http://localhost:8000

### 4. API 接口说明

#### 健康检查
```bash
GET http://localhost:8000/health
```

#### 单文件 OCR
```bash
POST http://localhost:8000/ocr
Content-Type: multipart/form-data

file: (图片文件)
prompt: (可选) 识别提示词
```

**Python 调用：**
```python
from ocr_client import GLMOCRClient

client = GLMOCRClient("http://localhost:8000")
result = client.recognize_file("image.jpg")
print(result['text'])
```

#### Base64 图片 OCR
```bash
POST http://localhost:8000/ocr/base64
Content-Type: application/json

{
    "image_base64": "base64字符串",
    "prompt": "识别图片中的文字"
}
```

**Python 调用：**
```python
from ocr_client import GLMOCRClient, image_to_base64

client = GLMOCRClient("http://localhost:8000")
base64_str = image_to_base64("image.jpg")
result = client.recognize_base64(base64_str)
```

## 本地使用（不启动服务器）

如果不需要 HTTP 服务，可以直接使用本地脚本：

```python
from ocr_local import GLMOCRProcessor

# 初始化
processor = GLMOCRProcessor(model_path="THUDM/glm-4-9b")
processor.load()

# 单张识别
result = processor.recognize("image.jpg")
print(result)

# 批量识别
results = processor.batch_recognize(["1.jpg", "2.jpg", "3.jpg"])
```

或命令行使用：

```bash
# 单文件
python ocr_local.py image.jpg

# 批量
python ocr_local.py 1.jpg 2.jpg 3.jpg -o result.txt

# 4bit量化（省显存）
python ocr_local.py image.jpg --4bit
```

## 模型配置（重要）

### 修改模型缓存路径

默认缓存路径为 `D:\models\huggingface`，如需修改：

```bash
# Windows CMD
set HF_HOME=C:\你的\自定义\路径

# 或在代码中修改 ocr_server.py / ocr_local.py 开头部分
os.environ.setdefault('HF_HOME', r'D:\models\huggingface')
```

### 选择适合的模型

根据你的显存选择模型：

| 模型 | 参数 | FP16 显存 | 4-bit 显存 | 推荐度 |
|------|------|-----------|------------|--------|
| `openbmb/MiniCPM-V-2_6` | 4B | ~10GB | ~4GB | ⭐⭐⭐⭐⭐ (推荐) |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | ~16GB | ~5GB | ⭐⭐⭐⭐ |
| `THUDM/glm-4v-9b` | 9B | ~20GB | ~7GB | ⭐⭐⭐ |

**8GB 显存推荐 MiniCPM-V-2_6**，无需量化即可流畅运行。

修改模型（编辑代码或设置环境变量）：
```bash
# 使用环境变量
set GLM_OCR_MODEL=openbmb/MiniCPM-V-2_6
python ocr_server.py
```

### 显存优化

如果显存不足，使用 4-bit 量化：

```bash
# 启用 4-bit 量化
set LOAD_4BIT=true
python ocr_server.py
```

或代码中：
```python
from ocr_local import GLMOCRProcessor
processor = GLMOCRProcessor(load_4bit=True)  # 默认已开启
processor.load()
```

## 常见问题

### 1. CUDA 版本不匹配
```bash
# 查看当前 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果版本不对，重新安装对应版本
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. 模型下载慢
设置 HuggingFace 镜像：
```bash
set HF_ENDPOINT=https://hf-mirror.com
python ocr_server.py
```

### 3. 显存不足 (OOM)
- 启用 4-bit 量化：`load_4bit=True`
- 减小图片尺寸
- 关闭其他占用显存的程序

### 4. 端口被占用
修改 `ocr_server.py` 底部的端口：
```python
uvicorn.run("ocr_server:app", host="0.0.0.0", port=8080)  # 改为8080
```

## 进阶用法

### 自定义提示词

```python
# 基础识别
prompt = "识别图片中的文字"

# 保持格式
prompt = "识别图片中的所有文字，保持原有格式"

# 表格识别
prompt = "识别图片中的表格内容，输出为Markdown格式"

# 发票识别
prompt = "识别图片中的发票信息，提取关键字段"

# 名片识别
prompt = "识别图片中的名片信息，提取姓名、电话、邮箱"
```

### 集成到其他应用

```python
import requests

def ocr_image(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/ocr",
            files={"file": f},
            data={"prompt": "识别图片中的文字"}
        )
    return response.json()['text']

# 使用
text = ocr_image("document.jpg")
```

## 许可证

遵循 GLM-4 模型许可证
