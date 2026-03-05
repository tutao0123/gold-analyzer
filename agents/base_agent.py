"""
Lightweight LLM Agent base class, replacing agentscope.agents.DialogAgent.
Calls the large model directly via DashScope's OpenAI-compatible interface.
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
    Lightweight conversational agent base class.

    Subclasses override reply() to insert custom logic before/after calling
    the LLM; call super().reply(msg) when an LLM call is needed.
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
        """Call the LLM and return its text response."""
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
            logger.error(f"[{self.name}] LLM call failed: {e}", exc_info=True)
            raise

    def reply(self, x) -> Msg:
        """
        Base implementation: calls the LLM with x.content as the user message
        and returns a Msg. If x is a list of Msg objects, concatenates all
        contents before calling.
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
