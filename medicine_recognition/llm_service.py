"""
LLM service module for Qwen-VL-Max integration.
"""

import os
import json
import logging
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    dashscope = None
    MultiModalConversation = None

from config import DASHSCOPE_API_KEY, QWEN_MODEL

logger = logging.getLogger(__name__)


class LLMService:
    """Qwen-VL-Max service for medicine recognition."""

    _instance = None
    _api_key = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._api_key is None:
            self._api_key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. Please set it via environment variable."
                )
            if dashscope:
                dashscope.api_key = self._api_key
            self._model = QWEN_MODEL
            logger.info(f"LLM service initialized with model: {self._model}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _load_image_as_file(self, image_path: str) -> str:
        """
        Load image and return file path for the model.
        Qwen-VL supports both file paths and base64.

        Args:
            image_path: Path to the image file

        Returns:
            Path string that can be used by the model
        """
        return image_path

    def recognize_with_references(
        self,
        target_image_path: str,
        reference_samples: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Recognize medicine bottle using reference samples.

        Args:
            target_image_path: Path to the target image to recognize
            reference_samples: List of reference samples with image_path and text_label
            max_retries: Max number of retries on API failure

        Returns:
            Dict containing recognized medicine info
        """
        # Build messages
        messages = self._build_prompt(reference_samples, target_image_path)

        # Call API with retries
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling Qwen-VL-Max API (attempt {attempt + 1})")
                start_time = time.time()

                response = MultiModalConversation.call(
                    model=self._model,
                    messages=messages,
                )

                elapsed = time.time() - start_time
                logger.info(f"LLM API call completed in {elapsed:.2f}s")

                if response.status_code == 200:
                    # Parse response
                    result = self._parse_response(response)
                    return result
                else:
                    logger.error(f"API error: {response.code} - {response.message}")
                    if attempt < max_retries:
                        logger.info("Retrying...")
                        time.sleep(1)
                    else:
                        raise Exception(f"API error: {response.message}")

            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                if attempt < max_retries:
                    logger.info("Retrying...")
                    time.sleep(1)
                else:
                    raise

    def _build_prompt(
        self,
        reference_samples: List[Dict[str, Any]],
        target_image_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Build prompt with reference samples and target image.

        Args:
            reference_samples: List of reference samples
            target_image_path: Path to target image

        Returns:
            Messages list for MultiModalConversation
        """
        content = []

        # Add reference samples
        for i, sample in enumerate(reference_samples, 1):
            content.append({"text": f"参考样本{i}: "})
            # Use local file path - Qwen-VL-Max supports local files
            content.append({"image": sample["image_path"]})
            content.append({"text": f"标签: {sample['text_label']}"})

        # Add target image
        content.append({"text": "待识别图片: "})
        content.append({"image": target_image_path})

        # Add instruction
        content.append({
            "text": "请根据参考样本识别这个药瓶，输出 JSON 格式：{\"name\":\"药名\", \"specification\":\"规格\", "
            "\"manufacturer\":\"生产厂家\", \"usage\":\"用法用量\", \"warning\":\"注意事项\"}。只输出 JSON，不要有其他文字。"
        })

        messages = [{"role": "user", "content": content}]

        return messages

    def _parse_response(self, response) -> Dict[str, Any]:
        """
        Parse the LLM response to extract medicine info.

        Args:
            response: Response from MultiModalConversation

        Returns:
            Dict containing medicine information
        """
        try:
            content = response.output.choices[0].message.content
            text_content = ""

            # Handle both list and string content format
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        text_content += item["text"]
                    elif isinstance(item, str):
                        text_content += item
            else:
                text_content = content

            # Try to parse as JSON
            # Find JSON block in the response
            json_start = text_content.find("{")
            json_end = text_content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text_content[json_start:json_end]
                medicine_info = json.loads(json_str)
            else:
                # Try parsing the whole response
                medicine_info = json.loads(text_content)

            return {
                "medicine": medicine_info,
                "raw_response": text_content,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response content: {text_content}")
            return {
                "medicine": {
                    "name": "",
                    "specification": "",
                    "manufacturer": "",
                    "usage": "",
                    "warning": "",
                },
                "raw_response": text_content,
                "parse_error": str(e),
            }
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise


def call_llm(
    target_image_path: str,
    reference_samples: List[Dict[str, Any]],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function to call LLM service.

    Args:
        target_image_path: Path to target image
        reference_samples: List of reference samples
        max_retries: Max retries

    Returns:
        Medicine info dict
    """
    service = LLMService()
    return service.recognize_with_references(
        target_image_path, reference_samples, max_retries
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test if API key is set
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        print("Please set it with: export DASHSCOPE_API_KEY=your_api_key")
        sys.exit(1)

    print("LLM service test - API key is configured")
    print(f"Model: {QWEN_MODEL}")