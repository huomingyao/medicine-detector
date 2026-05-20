# 医院配液中心药品智能识别系统

<p align="center">
  <a href="https://github.com/huomingyao/medicine-detector/stargazers"><img src="https://img.shields.io/github/stars/huomingyao/medicine-detector" alt="Stars"></a>
  <a href="https://github.com/huomingyao/medicine-detector/issues"><img src="https://img.shields.io/github/issues/huomingyao/medicine-detector" alt="Issues"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
</p>

本项目提供医院配液中心药品智能识别的多种技术方案实现。

## 技术方案

本项目实现了三种不同的药品识别技术方案：

| 分支 | 技术方案 | 描述 |
|------|---------|------|
| [yolo-direct](tree/yolo-direct) | YOLO直接识别 | 直接训练YOLO模型识别药品类别 |
| [yolo+ocr](tree/yolo+ocr) | YOLO+OCR+LLM | YOLO检测药瓶 → OCR识别文字 → LLM推理 |
| [multimodal](tree/multimodal) | 向量库+多模态 | CLIP视觉语义匹配 + 向量检索 |

### 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|-----|------|----------|
| YOLO直接识别 | 速度快，无需OCR | 需要大量标注数据 | 药品包装标准化 |
| YOLO+OCR+LLM | 精度高，可处理模糊文字 | 依赖OCR和LLM API | 文字清晰的药瓶 |
| 向量库+多模态 | 无需训练，可扩展 | 依赖向量库 | 新药品快速上线 |

### 详细文档

- [YOLO直接识别方案](tree/yolo-direct) - 直接训练YOLO模型识别药品
- [YOLO+OCR+LLM方案](tree/yolo+ocr) - 结合OCR和LLM的方案
- [多模态方案](tree/multimodal) - 使用CLIP和向量库的方案

## 快速开始

### 克隆项目

```bash
git clone https://github.com/huomingyao/medicine-detector.git
cd medicine-detector

# 查看所有分支
git branch -a

# 切换到指定方案
git checkout yolo+ocr  # 或 yolo-direct, multimodal
```

### 选择技术方案

根据你的场景选择合适的分支：

1. **如果药品包装统一、数据充足** → 使用 `yolo-direct` 分支
2. **如果需要识别文字、数据有限** → 使用 `yolo+ocr` 分支  
3. **如果需要快速上线新药品** → 使用 `multimodal` 分支

## 项目结构

```
medicine-detector/
├── src/                    # 源代码
├── yolo-direct/           # YOLO直接识别方案
├── yolo+ocr/             # YOLO+OCR+LLM方案
├── multimodal/           # 多模态方案
└── docs/                # 文档
```

## 许可证

MIT License