#!/usr/bin/env python
"""
药瓶文字库建立脚本
使用千问多模态模型API提取完整药瓶标签信息

流程：
1. 第1轮：识别所有药品
2. 第2轮：识别所有药品
3. 第3轮：识别所有药品
4. 用阿里云百炼 deepseek-r1 整合每种药品的3次结果

环境变量:
  DASHSCOPE_API_KEY: 千问API密钥（多模态模型）+ deepseek-r1（文本模型）
"""
import os
import sys
import json
import time
import base64
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# ==================== 千问多模态API调用 ====================

def call_qwen_vision_once(api_key: str, image_paths: List[str], prompt: str = None, model: str = "qwen-vl-plus-latest", max_retries: int = 3) -> str:
    """单次调用多模态模型"""
    import requests

    if prompt is None:
        prompt = """请仔细观察这几张药瓶图片，将标签上所有能看到的文字全部抄录下来。

要求：
1. 不要遗漏任何文字，包括中文、英文、数字、符号
2. 从上到下依次抄录，保持阅读顺序
3. 瓶身、瓶盖、瓶签上的文字都要包含
4. 只输出文字内容，不要添加任何解释或标注"""

    # 转换图片为base64
    images_content = []
    for img_path in image_paths[:4]:  # 最多4张
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
            images_content.append({"image": f"data:image/jpeg;base64,{img_base64}"})

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    content = images_content.copy()
    content.append({"text": prompt})

    data = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 2048
        }
    }

    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            result = response.json()

            if response.status_code != 200:
                if retry < max_retries - 1:
                    time.sleep(2)
                continue

            output = result.get('output', {})
            text = ''
            if 'text' in output and output['text']:
                text = output['text']
            elif 'choices' in output and output['choices']:
                choice = output['choices'][0]
                if 'message' in choice:
                    msg = choice['message']
                    content = msg.get('content', '')
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get('text', '') or str(content)
                    elif content:
                        text = str(content)
            elif 'generated_text' in output:
                text = output['generated_text']

            if text:
                return text.strip()

        except Exception:
            if retry < max_retries - 1:
                time.sleep(2)
            continue

    return ""


def get_dashscope_api_key() -> str:
    """获取阿里云百炼 API 密钥，优先使用 DASHSCOPE_API_KEY"""
    return os.environ.get('DASHSCOPE_API_KEY', '')


def merge_single_drug_ocr(result1: str, result2: str, result3: str) -> str:
    """使用阿里云百炼 deepseek-r1 整合单个药品的3次OCR结果"""
    import requests

    api_key = get_dashscope_api_key()
    if not api_key:
        # 如果没有 API 密钥，直接合并所有结果
        return result1 + "\n" + result2 + "\n" + result3

    prompt = f"""我有三份来自不同识别轮次的药瓶标签文字内容，可能各有遗漏或错误。

请整合这三份内容，取长补短，输出最完整、最准确的标签文字。

识别结果1:
{result1}

识别结果2:
{result2}

识别结果3:
{result3}

请按从上到下的顺序输出整合后的完整标签文字，不要添加任何解释。"""

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-r1",
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 4096
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        result = response.json()

        if response.status_code == 200:
            output = result.get('output', {})
            choices = output.get('choices', [])
            if choices:
                return choices[0]['message']['content']
    except Exception:
        pass

    # 如果失败，直接合并
    return result1 + "\n" + result2 + "\n" + result3


def parse_vision_result(text: str) -> Dict[str, str]:
    """解析多模态模型返回的结果"""
    import re

    result = {
        'all_text': '',
    }

    if not text:
        return result

    # 清理一些可能的标注前缀
    text = text.strip()
    text = re.sub(r'^(以下是标签文本：|标签内容：|识别结果：|提取的文字：|整合后：|整合结果：)\s*', '', text, flags=re.IGNORECASE)

    result['all_text'] = text
    return result


# ==================== 文字库建立 ====================

class DrugLibraryBuilder:
    """药瓶文字库建立器 - 使用千问多模态模型"""

    def __init__(self, library_dir: str = "drug_library", output_file: str = "drug_library.json"):
        self.library_dir = Path(library_dir)
        self.output_file = output_file

        # 从环境变量获取API密钥（统一使用 DASHSCOPE_API_KEY）
        self.api_key = get_dashscope_api_key()

        if not self.api_key:
            print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
            print("请设置: set DASHSCOPE_API_KEY=你的阿里云百炼API密钥")
        else:
            print(f"阿里云百炼API密钥已设置: {self.api_key[:10]}...")
            print("将使用 deepseek-r1 整合3次识别结果")

        # 模型选择
        self.model = 'qwen-vl-plus-latest'
        print(f"使用模型: {self.model}")

    def scan_library_directory(self) -> dict:
        """扫描文字库目录"""
        drug_images = {}

        if not self.library_dir.exists():
            print(f"文字库目录不存在: {self.library_dir}")
            return drug_images

        for drug_dir in self.library_dir.iterdir():
            if drug_dir.is_dir():
                drug_name = drug_dir.name
                images = []

                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    images.extend(drug_dir.glob(ext))
                    images.extend(drug_dir.glob(ext.upper()))

                if images:
                    unique_images = list(set([str(p) for p in images]))[:4]
                    drug_images[drug_name] = unique_images
                    print(f"  发现药瓶: {drug_name} ({len(unique_images)} 张照片)")

        return drug_images

    def build_library(self) -> dict:
        """建立文字库 - 3轮识别 + LLM整合"""
        if not self.api_key:
            print("错误: 请先设置 DASHSCOPE_API_KEY 环境变量")
            print("获取API密钥: https://dashscope.console.aliyun.com/")
            return {}

        drug_images = self.scan_library_directory()

        if not drug_images:
            print("\n未找到药瓶图片，请按以下结构创建目录:")
            print("  drug_library/")
            print("    药瓶名称1/")
            print("      photo1.jpg")
            print("      photo2.jpg")
            print("    药瓶名称2/")
            print("    ...")
            return {}

        # 存储每种药品3次识别结果
        all_round_results = {drug: [] for drug in drug_images}

        drug_list = list(drug_images.keys())
        total = len(drug_list)

        # 第1轮：识别所有药品
        print(f"\n========== 第1轮识别 ({total}种药品) ==========")
        for i, drug_name in enumerate(drug_list):
            print(f"[{i+1}/{total}] {drug_name}...", end=" ", flush=True)
            result = call_qwen_vision_once(self.api_key, drug_images[drug_name], model=self.model)
            all_round_results[drug_name].append(result)
            print(f"完成 ({len(result)} 字符)")

        # 第2轮：识别所有药品
        print(f"\n========== 第2轮识别 ({total}种药品) ==========")
        for i, drug_name in enumerate(drug_list):
            print(f"[{i+1}/{total}] {drug_name}...", end=" ", flush=True)
            result = call_qwen_vision_once(self.api_key, drug_images[drug_name], model=self.model)
            all_round_results[drug_name].append(result)
            print(f"完成 ({len(result)} 字符)")

        # 第3轮：识别所有药品
        print(f"\n========== 第3轮识别 ({total}种药品) ==========")
        for i, drug_name in enumerate(drug_list):
            print(f"[{i+1}/{total}] {drug_name}...", end=" ", flush=True)
            result = call_qwen_vision_once(self.api_key, drug_images[drug_name], model=self.model)
            all_round_results[drug_name].append(result)
            print(f"完成 ({len(result)} 字符)")

        # 用LLM整合每种药品的3次结果
        print(f"\n========== LLM整合 ==========")
        library = {}
        for i, drug_name in enumerate(drug_list):
            print(f"[{i+1}/{total}] 整合 {drug_name}...", end=" ", flush=True)
            results = all_round_results[drug_name]
            combined = merge_single_drug_ocr(results[0], results[1], results[2])
            parsed = parse_vision_result(combined)

            library[drug_name] = {
                'images': [os.path.basename(p) for p in drug_images[drug_name]],
                'all_text': parsed.get('all_text', '')
            }

            print(f"完成 ({len(library[drug_name]['all_text'])} 字符)")

        return library

    def save_library(self, library: dict):
        """保存文字库"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, ensure_ascii=False, indent=2)

        print(f"\n文字库已保存: {output_path}")

    def create_demo_directory(self):
        """创建示例目录"""
        self.library_dir.mkdir(parents=True, exist_ok=True)

        examples = ['氨基己酸注射液', '氯化钠注射液', '葡萄糖注射液']

        for example in examples:
            example_dir = self.library_dir / example
            example_dir.mkdir(parents=True, exist_ok=True)

            readme = example_dir / "说明.txt"
            with open(readme, 'w', encoding='utf-8') as f:
                f.write(f"请将「{example}」的照片放入此目录\n")
                f.write("建议拍摄: 正面、侧面、顶部等不同角度\n")
                f.write("支持格式: jpg, jpeg, png, bmp, tiff\n")

        print(f"示例目录已创建: {self.library_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='药瓶文字库建立工具')
    parser.add_argument('--dir', '-d', type=str, default='drug_library',
                        help='药瓶图片目录 (默认: drug_library)')
    parser.add_argument('--output', '-o', type=str, default='drug_library.json',
                        help='输出文件 (默认: drug_library.json)')
    parser.add_argument('--demo', action='store_true',
                        help='创建示例目录')

    args = parser.parse_args()

    builder = DrugLibraryBuilder(
        library_dir=args.dir,
        output_file=args.output
    )

    if args.demo:
        builder.create_demo_directory()
        return

    library = builder.build_library()
    if library:
        builder.save_library(library)
        print(f"\n成功处理 {len(library)} 种药品")


if __name__ == "__main__":
    main()