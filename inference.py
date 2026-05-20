#!/usr/bin/env python
"""
药品识别推理脚本
直接运行脚本进行识别，无需命令行参数

修改下面的配置来更改设置:
"""
import os
import sys
import json
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置区域 ====================
# 在这里直接修改配置
YOLO_MODEL_PATH = "d:/ocr+yolo代码 - 副本/yolov8x-v8.2.0.pt"  # YOLO模型路径
DRUG_LIST_PATH = "d:/ocr+yolo代码 - 副本/classes.txt"  # 药品列表
LIBRARY_PATH = "d:/ocr+yolo代码 - 副本/drug_library.json"  # 文字库(可选)

# 图片路径 - 修改这里指定要识别的图片
IMAGE_PATH = "D:/ocr+yolo代码 - 副本/test"

# 输出结果目录和文件
OUTPUT_DIR = "D:/ocr_results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.json")


# 识别参数
USE_OCR = True  # 是否使用OCR
USE_LLM = True  # 是否使用LLM推理
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TARGET_CLASSES = ["bottle"]  # 只识别 bottle 类别
# ==================== 配置结束 ====================

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.drug_recognizer import DrugRecognizer
from src.text_matcher import load_drug_list_from_file


def load_images_from_dir(input_dir: str) -> List[str]:
    """从目录加载图片列表（去重）"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []

    input_path = Path(input_dir)
    if input_path.is_dir():
        for ext in image_extensions:
            images.extend(input_path.glob(f'*{ext}'))
            images.extend(input_path.glob(f'*{ext.upper()}'))
    elif input_path.is_file() and input_path.suffix == '.txt':
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    images.append(line)

    # 去重
    seen = set()
    unique_images = []
    for img in images:
        img_str = str(img) if isinstance(img, Path) else img
        if img_str not in seen:
            seen.add(img_str)
            unique_images.append(img_str)

    return unique_images


def save_results(results: list, output_path: str):
    """保存结果到JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.replace('\n', ' ').split(' ')
    lines = []
    current = ''
    for word in words:
        if not word:
            continue
        candidate = f"{current} {word}".strip()
        # 使用 getlength 替代已弃用的 getsize (Pillow 10+)
        width = font.getlength(candidate)
        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_label(draw: ImageDraw.Draw, position: tuple[int, int], text: str, font: ImageFont.ImageFont):
    if not text:
        return
    x, y = position
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    padding = 10
    rect = [x, y, x + text_width + padding * 2, y + text_height + padding * 2]
    draw.rectangle(rect, fill=(0, 0, 0, 200))
    draw.text((x + padding, y + padding), text, fill=(255, 255, 255), font=font)


def _get_chinese_font(size: int = 16) -> ImageFont.ImageFont:
    """获取支持中文的字体"""
    import platform
    import os

    font_names = [
        'msyh.ttc',  # 微软雅黑
        'simhei.ttf',  # 黑体
        'simsun.ttc', # 宋体
        'Microsoft YaHei.ttf',
        'SimHei.ttf',
    ]

    # Windows系统从Fonts目录查找
    if platform.system() == 'Windows':
        fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        if os.path.exists(fonts_dir):
            for font_name in font_names:
                font_path = os.path.join(fonts_dir, font_name)
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue

    # 尝试当前目录
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue

    # 找不到中文字体时使用默认
    return ImageFont.load_default()


def save_annotated_image(image_path: str, fused_results: list, save_path: str):
    """保存带标注的图片,只显示YOLO框+匹配结果"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = _get_chinese_font(50)

    for fused in fused_results:
        if not fused.bbox:
            continue
        x1, y1, x2, y2 = fused.bbox

        # 绘制加粗检测框 (双层框)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=6)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # 只显示药品名称
        llm_drug = fused.llm_raw_response or fused.final_drug or "未知"
        _draw_label(draw, (x1, max(0, y1 - 60)), llm_drug, font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"已保存标注图片: {save_path}")


def save_ocr_crop_visuals(image_path: str, fused_results: list, output_dir: str):
    img = Image.open(image_path).convert('RGB')
    os.makedirs(output_dir, exist_ok=True)
    font = _get_chinese_font(14)

    for idx, fused in enumerate(fused_results, start=1):
        if not fused.bbox:
            continue
        x1, y1, x2, y2 = fused.bbox
        crop = img.crop((x1, y1, x2, y2))
        draw = ImageDraw.Draw(crop)
        ocr_text = fused.ocr_text or fused.ocr_raw_text or 'OCR未识别到内容'
        lines = _wrap_text(ocr_text, font, crop.width - 12)
        if lines:
            # 使用 getbbox 替代已弃用的 getsize (Pillow 10+)
            _, _, _, text_h = font.getbbox(lines[0])
            text_h = int(text_h)
            text_height = sum(int(font.getlength(line)) for line in lines) + 8 * len(lines)
            overlay_height = text_height + 10
            # 确保 overlay 尺寸与 crop 一致
            crop_rgba = crop.convert('RGBA')
            overlay = Image.new('RGBA', crop_rgba.size, (0, 0, 0, 160))
            # 调整 overlay 高度并粘贴到 crop 底部
            if overlay_height < crop_rgba.height:
                overlay = overlay.crop((0, 0, crop_rgba.width, overlay_height))
            crop_rgba.paste(overlay, (0, crop_rgba.height - overlay_height), overlay)
            crop = crop_rgba
            draw = ImageDraw.Draw(crop)
            y_text = crop.height - overlay_height + 5
            for line in lines:
                draw.text((6, y_text), line, fill=(255, 255, 255), font=font)
                y_text += text_h + 4
        crop = crop.convert('RGB')
        crop_path = os.path.join(output_dir, f"{Path(image_path).stem}_crop_{idx}_ocr.jpg")
        crop.save(crop_path)


def save_detection_details(image_path: str, fused_results: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"图片: {image_path}\n")
        f.write(f"检测目标数量: {len(fused_results)}\n\n")
        for idx, fused in enumerate(fused_results, start=1):
            f.write(f"目标 {idx}:\n")
            f.write(f"  位置: {fused.bbox}\n")
            f.write(f"  最终结果: {fused.final_drug}\n")
            f.write(f"  决策: {fused.decision}\n")
            f.write(f"  来源: {fused.source}\n")
            f.write(f"  YOLO置信度: {fused.yolo_conf:.3f}\n")
            f.write(f"  OCR置信度: {fused.ocr_conf:.3f}\n")
            f.write(f"  OCR匹配药品: {fused.ocr_drug}\n")
            f.write(f"  OCR文本: {fused.ocr_text or fused.ocr_raw_text}\n")
            if fused.llm_raw_response:
                f.write(f"  LLM原始返回: {fused.llm_raw_response}\n")
            f.write("\n")


def main():
    """主函数 - 直接运行"""
    # 检查YOLO模型
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"错误: YOLO模型不存在: {YOLO_MODEL_PATH}")
        return 1

    # 加载药品列表或文字库
    drug_list = []
    use_library = False
    library_path = None

    if LIBRARY_PATH and os.path.exists(LIBRARY_PATH):
        # 使用文字库模式
        use_library = True
        library_path = LIBRARY_PATH
        print(f"使用文字库模式: {LIBRARY_PATH}")
        print(f"LLM推理: {'启用' if USE_LLM else '禁用'}")
        print(f"OCR识别: {'启用' if USE_OCR else '禁用'}")
    else:
        # 使用药品列表模式
        if not os.path.exists(DRUG_LIST_PATH):
            print(f"错误: 药品列表不存在: {DRUG_LIST_PATH}")
            return 1

        drug_list = load_drug_list_from_file(DRUG_LIST_PATH)
        if not drug_list:
            print(f"错误: 药品列表为空: {DRUG_LIST_PATH}")
            return 1
        print(f"使用药品列表模式: {len(drug_list)} 种药品")

    # 创建识别器
    recognizer = DrugRecognizer(
        yolo_model_path=YOLO_MODEL_PATH,
        drug_list=drug_list,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        library_path=library_path,
        use_llm=use_library and USE_LLM,
        target_classes=TARGET_CLASSES
    )

    print(f"OCR模式: {'GLM-OCR' if USE_OCR else '关闭'}")

    # 收集图片
    images = []
    if os.path.isfile(IMAGE_PATH):
        images = [IMAGE_PATH]
    elif os.path.isdir(IMAGE_PATH):
        images = load_images_from_dir(IMAGE_PATH)
    else:
        print(f"错误: 图片路径不存在: {IMAGE_PATH}")
        return 1

    if not images:
        print("错误: 未找到图片")
        return 1

    print(f"找到 {len(images)} 张图片")

    # 识别
    all_results = []
    for img_path in images:
        print(f"\n[{images.index(img_path)+1}/{len(images)}] 识别: {img_path}")
        try:
            result = recognizer.recognize(img_path, use_ocr=USE_OCR)

            detections = []
            for fused in result.fused_results:
                detections.append({
                    'drug': fused.final_drug,
                    'confidence': round(fused.confidence, 3),
                    'decision': fused.decision,
                    'source': fused.source,
                    'yolo_conf': round(fused.yolo_conf, 3),
                    'ocr_conf': round(fused.ocr_conf, 3),
                })

            image_dir = os.path.join(OUTPUT_DIR, Path(img_path).stem)
            os.makedirs(image_dir, exist_ok=True)

            annotated_path = os.path.join(image_dir, f"{Path(img_path).stem}_annotated.jpg")
            save_annotated_image(img_path, result.fused_results, annotated_path)

            crop_dir = os.path.join(image_dir, "ocr_crops")
            save_ocr_crop_visuals(img_path, result.fused_results, crop_dir)

            detail_path = os.path.join(image_dir, f"{Path(img_path).stem}_details.txt")
            save_detection_details(img_path, result.fused_results, detail_path)

            output = {
                'image_path': img_path,
                'detection_count': len(detections),
                'detections': detections,
                'annotated_image': annotated_path,
                'ocr_crop_dir': crop_dir,
                'detail_file': detail_path,
                'time': {
                    'yolo': round(result.yolo_time, 2),
                    'ocr': round(result.ocr_time, 2),
                    'llm': round(result.llm_time, 2),
                    'total': round(result.total_time, 2),
                }
            }

            all_results.append(output)

            # 打印结果
            print(f"\n图片: {img_path}")
            print(f"检测到 {len(result.fused_results)} 个药品:")
            if len(result.fused_results) == 0:
                print("  (警告: 未检测到任何目标)")
                print(f"  YOLO检测数量: {len(result.detections)}")
            else:
                for fused in result.fused_results:
                    print(f"  - {fused.final_drug} (置信度: {fused.confidence:.2f}, 来源: {fused.decision}, 来源详情: {fused.source})")

            llm_time = result.llm_time
            if llm_time > 0:
                print(f"\n耗时: YOLO {result.yolo_time:.2f}s | OCR {result.ocr_time:.2f}s | LLM {llm_time:.2f}s | 总计 {result.total_time:.2f}s")
            else:
                print(f"\n耗时: YOLO {result.yolo_time:.2f}s | OCR {result.ocr_time:.2f}s | 总计 {result.total_time:.2f}s")

        except Exception as e:
            print(f"识别失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    if all_results:
        save_results(all_results, OUTPUT_PATH)
        print(f"\n结果已保存到: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())