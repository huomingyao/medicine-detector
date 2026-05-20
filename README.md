# 药瓶识别系统

基于向量检索与多模态大模型的药瓶识别 API，使用 Milvus + Chinese-CLIP + Qwen-VL-Max。

## 目录结构

```
medicine_recognition/
├── app.py              # FastAPI 主入口
├── config.py           # 配置文件
├── vectorizer.py       # Chinese-CLIP 向量化模块
├── llm_service.py     # Qwen-VL-Max 调用
├── yolo_detector.py   # YOLO 药瓶检测
├── simple_store.py    # 简单向量存储（备用）
├── models.py          # Pydantic 模型
├── requirements.txt   # Python 依赖
├── docker-compose.yml # Docker 部署配置
├── Dockerfile       # Docker 镜像配置
├── samples/        # 样本图片存放目录
├── uploads/        # 上传图片目录
├── crops/         # 裁剪图片目录
└── README.md      # 本文档
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- CPU: 推荐 8 核+
- 内存: 推荐 8GB+
- 磁盘: 10GB+
- GPU: 可选，用于加速 Chinese-CLIP 向量化

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

获取 API Key: [阿里云 DashScope](https://dashscope.aliyuncs.com/)

### 4. 启动服务

```bash
# 直接运行
python app.py

# 或使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000

# 使用 Docker
docker-compose up -d
```

### 5. 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 响应示例:
# {
#   "status": "healthy",
#   "milvus": {"status": "healthy", "samples": 0},
#   "llm": {"status": "available"}
# }
```

## API 接口

### 添加样本

```bash
curl -X POST http://localhost:8000/api/v1/samples \
  -F "file=@sample.jpg" \
  -F "text_label=阿莫西林胶囊 0.5g 白云山制药"
```

### 获取样本列表

```bash
curl http://localhost:8000/api/v1/samples

# 带关键词过滤
curl "http://localhost:8000/api/v1/samples?keyword=阿莫西林"
```

### 识别药瓶

```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "file=@target.jpg"
```

响应示例：

```json
{
  "success": true,
  "message": "识别成功",
  "medicine": {
    "name": "阿莫西林胶囊",
    "specification": "0.5g/粒",
    "manufacturer": "白云山制药股份有限公司",
    "usage": "成人一次0.5g，一日3次",
    "warning": "对青霉素过敏者禁用"
  },
  "matched_samples": [
    {
      "image_path": "/path/to/sample1.jpg",
      "label": "阿莫西林胶囊 0.5g 白云山制药",
      "similarity": 0.92
    }
  ]
}
```

## 技术原理

### 1. YOLO 检测药瓶

- 使用 YOLOv8 检测图片中的药瓶位置
- 返回每个药瓶的 bounding box 和置信度

### 2. Qwen-VL 多模态识别

- 对每个检测到的药瓶区域进行裁剪
- 调用 Qwen-VL-Max 多模态大模型识别药瓶上的文字
- 返回 JSON 格式的识别结果

### 3. 识别流程

```
上传图片 → YOLO 检测 → 裁剪药瓶 → Qwen-VL 识别 → 返回结果
```

## 配置说明

在 `config.py` 中可以修改：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| YOLO_MODEL_PATH | YOLO 模型路径 | `d:/api_shibie/yolov8x-v8.2.0.pt` |
| YOLO_CONFIDENCE | YOLO 置信度阈值 | `0.5` |
| DASHSCOPE_API_KEY | 阿里云 API Key | (环境变量) |
| API_PORT | API 端口 | `8000` |

## Docker 部署

### 使用预构建镜像

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/samples:/app/samples \
  -v $(pwd)/data:/app/data \
  -e DASHSCOPE_API_KEY=your_api_key \
  medicine-recognition:latest
```

### 构建镜像

```bash
docker build -t medicine-recognition .
```

### 使用 docker-compose

```bash
# 编辑 .env 文件设置 API key
echo "DASHSCOPE_API_KEY=your_key" > .env

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 常见问题

### 1. 模型下载慢

首次运行时会下载 Chinese-CLIP 模型（约 1GB），可使用国内镜像：

```python
# 在代码中指定 cache 目录
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

### 2. Milvus Lite vs Milvus

- **Milvus Lite**: 单文件存储，适合开发和测试
- **Milvus**: 完整版，需要 docker 部署，支持更多功能

如需切换到完整版 Milvus，修改配置：

```python
# config.py
MILVUS_URI = "http://localhost:19530"
```

### 3. 内存不足

如果内存不够，可以：
- 减少 batch size
- 使用 CPU 而非 GPU
- 使用 Milvus Lite（资源需求更低）

### 4. API 调用失败

检查：
- DASHSCOPE_API_KEY 是否正确
- 网络是否能访问阿里云
- API 配额是否用完

## 开发说明

### 添加新功能

1. 在 `app.py` 中添加新路由
2. 在 `models.py` 中定义请求/响应模型
3. 在相应模块中实现逻辑

### 运行测试

```bash
curl http://localhost:8000/health
```

### 日志级别

```python
# 修改日志级别
logging.basicConfig(level=logging.DEBUG)
```

## 许可证

MIT License