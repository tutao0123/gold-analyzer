"""
轻量级 LLM Agent 基类，替代 agentscope.agents.DialogAgent。
直接通过 DashScope 的 OpenAI 兼容接口调用大模型。
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional

import openai

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class Msg:
    name: str
    role: str
    content: str


class LLMAgent:
    """
    轻量级对话智能体基类。

    子类通过重写 reply() 在调用 LLM 前后插入自定义逻辑；
    需要调用 LLM 时执行 super().reply(msg)。
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = None,
        enable_search: bool = False,
        commodity: Optional[dict] = None,
    ):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model_name = model_name
        self.enable_search = enable_search
        self.commodity = commodity or {"key": "gold", "symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"}
        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=_DASHSCOPE_BASE_URL,
        )

    def _call_llm(self, user_content: str) -> str:
        """调用 LLM，返回文本回复。"""
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs: dict = dict(model=self.model_name, messages=messages)
        if self.enable_search:
            kwargs["extra_body"] = {"enable_search": True}
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}", exc_info=True)
            raise

    def reply(self, x) -> Msg:
        """
        基类实现：用 x.content 作为用户消息调用 LLM，返回 Msg。
        若 x 是 Msg 列表，则拼接所有内容后调用。
        """
        if x is None:
            return Msg(name=self.name, role="assistant", content="")

        if isinstance(x, list):
            content = "\n\n".join(
                f"【{m.name}】\n{m.content}" for m in x if isinstance(m, Msg)
            )
        elif isinstance(x, Msg):
            content = x.content
        else:
            content = str(x)

        result = self._call_llm(content)
        return Msg(name=self.name, role="assistant", content=result)

    def __call__(self, x) -> Msg:
        return self.reply(x)
