"""
DeepSeek API 封装

使用 OpenAI 兼容接口调用 DeepSeek，配置从 .env 读取。
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Generator

load_dotenv()


class DeepSeekLLM:
    """DeepSeek LLM 调用封装"""

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError(
                "未设置 DEEPSEEK_API_KEY\n"
                "请在 .env 文件中设置，或运行 export DEEPSEEK_API_KEY=sk-xxx"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stream: bool = False,
    ) -> str:
        """
        非流式对话（自动续接，防止 max_tokens 截断）

        Args:
            messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: 温度，越低越确定性
            max_tokens: 最大输出 token 数（单次）
            stream: 是否流式输出

        Returns:
            模型回复文本（完整，自动续接）
        """
        full_text = ""
        max_continues = 5  # 最多续接 5 次，防止死循环

        for _ in range(max_continues):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            choice = response.choices[0]
            full_text += choice.message.content

            # 如果是因为长度限制而停止，自动续接
            if choice.finish_reason == "length":
                # 将已生成的内容加入对话历史，让模型继续
                messages.append({"role": "assistant", "content": choice.message.content})
                messages.append({"role": "user", "content": "请继续输出，不要重复已经写过的内容。"})
                continue
            else:
                break

        return full_text

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """
        流式对话，逐块 yield 文本（自动续接，防止 max_tokens 截断）

        Yields:
            文本片段
        """
        max_continues = 5

        for _ in range(max_continues):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            full_text = ""
            finish_reason = None

            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_text += delta.content
                        yield delta.content
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

            # 如果是因为长度限制而停止，自动续接
            if finish_reason == "length":
                messages.append({"role": "assistant", "content": full_text})
                messages.append({"role": "user", "content": "请继续输出，不要重复已经写过的内容。"})
                continue
            else:
                break

    def _collect_stream(self, response) -> str:
        """收集流式响应为完整字符串"""
        full_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
        return full_text

    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        """带重试的对话"""
        for attempt in range(max_retries + 1):
            try:
                return self.chat(messages, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"  ️  API 调用失败（第{attempt + 1}次重试）: {e}")
                import time
                time.sleep(2 ** attempt)
