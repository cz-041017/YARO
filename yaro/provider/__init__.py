"""模型供应商适配层。

YARO 的 LLM 抽象：一个 LangChain ``BaseChatModel``，外部对 YARO 只知道"发消息、
收消息、带工具调用"，不知道底层是 DeepSeek / 内网 vLLM / 其它 OpenAI 兼容端点。

- :class:`~yaro.provider.deepseek.ChatDeepSeek` 是默认实现：DeepSeek 是 OpenAI
  兼容协议，所以直接继承 ``langchain_openai.ChatOpenAI``，只补 DeepSeek 特有的
  wire 层约束（reasoning_content 逐轮回传、图只能放 user 消息、content 列表规范化）。
- :func:`~yaro.provider.models.build_model` 按 ``Settings`` 构造模型实例。
- 出网脱敏（egress）**不在这里做**：它在调用模型之前、于图节点里对消息列表完成，
  这样"能否出网 / 能否发图"是 YARO 的统一策略，而不是某个供应商的私有实现。
"""

from .deepseek import ChatDeepSeek, deepseek_extra_body
from .models import build_model

__all__ = ["ChatDeepSeek", "build_model", "deepseek_extra_body"]
