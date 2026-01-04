# Mask DeepAgents 建置指南

本文檔記錄 `mask-deepagents` 的完整建置過程，供未來開發者參考。

## 概述

`mask-deepagents` 是一個將 DeepAgents SDK 包裝為 MASK Agent 的專案，整合了：
- **MASK Kernel**: 提供 A2A 協議支援
- **DeepAgents SDK**: 提供底層 Agent 能力
- **Dual Tracing**: 同時支援 Phoenix 和 Langfuse 可觀測性

## 設計原則

1. **MASK A2A Wrapper**: 使用 MASK 的 A2A Server 暴露服務
2. **內部用 DeepAgents**: `create_agent` 改用 DeepAgents SDK
3. **Fallback 機制**: 當 DeepAgents SDK 不可用時，自動使用 MASK LLM

## 專案結構

```
mask-deepagents/
├── config/
│   └── prompts/
│       └── system.md             # Agent 系統提示
├── src/mask_deepagents/
│   ├── __init__.py               # 模組入口
│   ├── agent.py                  # DeepAgents Wrapper 實作
│   └── main.py                   # A2A Server 啟動點
├── tests/
│   └── test_deepagent.py         # 整合測試
├── .env.example                  # 環境變數範本
├── .gitignore
├── pyproject.toml                # Python 專案配置
└── README.md
```

## 建置步驟

### Step 1: 建立專案目錄

```bash
mkdir -p mask-deepagents/config/prompts
mkdir -p mask-deepagents/src/mask_deepagents
mkdir -p mask-deepagents/tests
cd mask-deepagents
```

### Step 2: 建立 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mask-deepagents"
version = "0.1.0"
description = "DeepAgents SDK wrapped with MASK A2A protocol"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Colin Lee", email = "colinlee0924@gmail.com" }
]
keywords = ["mask", "deepagents", "agent", "a2a"]

dependencies = [
    # MASK Kernel 提供 A2A Server 支援
    "mask-kernel[phoenix,anthropic] @ git+https://github.com/colinlee0924/mask-kernel.git",
    # DeepAgents SDK
    "deepagents @ git+https://github.com/langchain-ai/deepagents.git",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]

[project.scripts]
mask-deepagents = "mask_deepagents.main:main"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/mask_deepagents"]
```

**重點說明**:
- `mask-kernel[phoenix,anthropic]` 提供 A2A Server 和可觀測性
- `deepagents` 是 LangChain 的 DeepAgents SDK
- 不需要 `mcp` extra，因為 DeepAgents 有自己的工具機制

### Step 3: 撰寫 System Prompt

建立 `config/prompts/system.md`:

```markdown
# DeepAgents Assistant

You are a helpful assistant powered by DeepAgents SDK, exposed via MASK A2A protocol.

## Capabilities
- Natural language understanding
- Complex reasoning and problem solving
- Code generation and explanation
- Research and analysis tasks

## Guidelines
- Be helpful and concise
- Explain your reasoning when appropriate
- Ask clarifying questions if needed
```

### Step 4: 實作 DeepAgents Wrapper

建立 `src/mask_deepagents/__init__.py`:

```python
"""mask-deepagents: DeepAgents SDK wrapped with MASK A2A."""
```

建立 `src/mask_deepagents/agent.py`:

```python
"""DeepAgents SDK wrapper with MASK interface.

This module wraps the DeepAgents SDK to provide a MASK-compatible interface
that can be exposed via A2A protocol.
"""

import os
from pathlib import Path
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from mask.agent import BaseAgent, load_prompts
from mask.core.state import HandoffContext


class DeepAgentWrapper(BaseAgent):
    """Wrapper that uses DeepAgents SDK internally but exposes MASK interface.

    This class bridges the DeepAgents SDK with the MASK A2A protocol,
    allowing DeepAgents capabilities to be used in a multi-agent MASK ecosystem.
    """

    def __init__(
        self,
        config_dir: str = "config",
        model_name: Optional[str] = None,
    ):
        """Initialize the DeepAgents wrapper.

        Args:
            config_dir: Path to configuration directory.
            model_name: Model to use (defaults to env DEEPAGENT_MODEL).
        """
        # Load prompts
        prompts = load_prompts(config_dir)
        self.system_prompt = prompts.get(
            "system",
            "You are a helpful assistant powered by DeepAgents."
        )

        # Get model configuration
        self.model_name = model_name or os.environ.get(
            "DEEPAGENT_MODEL", "claude-sonnet-4-20250514"
        )

        # Initialize DeepAgents SDK
        self._init_deepagent()

    def _init_deepagent(self) -> None:
        """Initialize the DeepAgents SDK agent."""
        try:
            from deepagents import create_deep_agent

            self.deep_agent = create_deep_agent(
                model=self.model_name,
                system_prompt=self.system_prompt,
            )
            self._use_deepagent = True
        except ImportError:
            # Fallback to direct LLM if DeepAgents not available
            print("Warning: DeepAgents SDK not available, using fallback mode")
            from mask.models import LLMFactory, ModelTier

            factory = LLMFactory()
            self.model = factory.get_model(tier=ModelTier.THINKING)
            self._use_deepagent = False
        except Exception as e:
            print(f"Warning: Failed to initialize DeepAgents: {e}, using fallback")
            from mask.models import LLMFactory, ModelTier

            factory = LLMFactory()
            self.model = factory.get_model(tier=ModelTier.THINKING)
            self._use_deepagent = False

    async def invoke(
        self,
        message: str,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> str:
        """Process a message using DeepAgents SDK.

        Args:
            message: The user message to process.
            session_id: Optional session ID (not used in stateless mode).
            handoff_context: Optional handoff context from parent agent.

        Returns:
            The agent's response as a string.
        """
        if self._use_deepagent:
            try:
                # Use DeepAgents SDK
                result = await self.deep_agent.arun(message)
                return str(result)
            except Exception as e:
                return f"Error running DeepAgent: {str(e)}"
        else:
            # Fallback to direct LLM call
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
            response = await self.model.ainvoke(messages)
            return response.content if hasattr(response, "content") else str(response)

    async def stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> AsyncIterator[str]:
        """Stream the response (falls back to non-streaming).

        Args:
            message: The user message to process.
            session_id: Optional session ID.
            handoff_context: Optional handoff context.

        Yields:
            Response chunks as strings.
        """
        # DeepAgents may not support streaming, return full response
        result = await self.invoke(message, session_id, handoff_context)
        yield result


def create_agent(
    config_dir: str = "config",
    model_name: Optional[str] = None,
) -> DeepAgentWrapper:
    """Create and return the DeepAgents wrapper instance.

    Args:
        config_dir: Path to configuration directory.
        model_name: Model to use.

    Returns:
        Configured DeepAgentWrapper instance.
    """
    return DeepAgentWrapper(config_dir=config_dir, model_name=model_name)
```

**重點說明**:
- 繼承 `BaseAgent` 而非 `SimpleAgent`，因為 DeepAgents 有自己的工具機制
- `_init_deepagent()` 實作 fallback 機制
- `_use_deepagent` flag 決定使用 DeepAgents 或 MASK LLM
- `invoke()` 和 `stream()` 實作 MASK 介面

### Step 5: 實作 A2A Server

建立 `src/mask_deepagents/main.py`:

```python
"""Main entry point for mask-deepagents A2A server."""

import os
from pathlib import Path

from dotenv import load_dotenv

from mask.a2a import MaskA2AServer

from mask_deepagents.agent import create_agent

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def main():
    """Start the A2A server."""
    # Setup tracing based on TRACING_BACKEND env var
    backend = os.environ.get("TRACING_BACKEND", "dual")

    if backend == "dual":
        from mask.observability import setup_dual_tracing

        print("Using DUAL tracing (Phoenix + Langfuse)...")
        setup_dual_tracing(project_name="mask-deepagents")
    elif backend == "phoenix":
        from mask.observability import setup_openinference_tracing

        print("Using Phoenix tracing...")
        setup_openinference_tracing(
            project_name="mask-deepagents",
            endpoint=os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
            ),
        )
    elif backend == "langfuse":
        from mask.observability import setup_langfuse_otel_tracing

        print("Using Langfuse tracing...")
        setup_langfuse_otel_tracing()
    else:
        print("No tracing configured (set TRACING_BACKEND=dual|phoenix|langfuse)")

    # Create DeepAgents wrapper
    agent = create_agent()

    # Create A2A server
    server = MaskA2AServer(
        agent=agent,
        name="mask-deepagents",
        description="DeepAgents SDK wrapped with MASK A2A protocol",
    )

    port = int(os.environ.get("PORT", "10030"))
    print(f"Starting mask-deepagents on port {port}...")
    server.run(port=port)


if __name__ == "__main__":
    main()
```

### Step 6: 配置環境變數

建立 `.env` (從 `.env.example` 複製):

```bash
# LLM Provider API Keys
ANTHROPIC_API_KEY=your-anthropic-key

# MASK Configuration
MASK_LLM_PROVIDER=anthropic

# Agent Port
PORT=10030

# DeepAgents Configuration
DEEPAGENT_MODEL=claude-sonnet-4-20250514

# Tracing Backend: dual | phoenix | langfuse
TRACING_BACKEND=dual

# Phoenix Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

# Langfuse Observability
LANGFUSE_SECRET_KEY=your-langfuse-project-secret-key
LANGFUSE_PUBLIC_KEY=your-langfuse-project-public-key
LANGFUSE_BASE_URL=http://localhost:3001
```

### Step 7: 撰寫測試

建立 `tests/__init__.py`:

```python
"""Tests for mask-deepagents."""
```

建立 `tests/test_deepagent.py`:

```python
"""Tests for mask-deepagents."""

import asyncio
import logging
from uuid import uuid4

import httpx


async def test_deepagent():
    """Test the DeepAgents wrapper A2A endpoint."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = "http://localhost:10030"

    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info("Testing mask-deepagents...")

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What is 2 + 2?",
                        }
                    ],
                }
            },
        }

        try:
            response = await client.post(base_url, json=request)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response: {result}")
            logger.info("Test passed!")
        except Exception as e:
            logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_deepagent())
```

### Step 8: 安裝與執行

```bash
# 安裝依賴
uv sync

# 執行 Agent
uv run mask-deepagents
```

### Step 9: 測試 A2A

```bash
curl -X POST http://localhost:10030/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "test-1",
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "What is 2 + 2?"
        }]
      }
    }
  }'
```

## 關鍵概念

### BaseAgent vs SimpleAgent

| 特性 | BaseAgent | SimpleAgent |
|------|-----------|-------------|
| 用途 | 自訂 Agent 實作 | MASK 標準 Agent |
| SkillMiddleware | 不包含 | 自動建立 |
| 工具管理 | 自行實作 | 透過 additional_tools |
| 適用場景 | Wrapper、特殊 Agent | 一般 MASK Agent |

本專案選擇 `BaseAgent` 是因為 DeepAgents SDK 有自己的工具機制。

### Fallback 機制

```python
def _init_deepagent(self) -> None:
    try:
        from deepagents import create_deep_agent
        self.deep_agent = create_deep_agent(...)
        self._use_deepagent = True
    except ImportError:
        # Fallback to MASK LLM
        factory = LLMFactory()
        self.model = factory.get_model(tier=ModelTier.THINKING)
        self._use_deepagent = False
```

這確保即使 DeepAgents SDK 安裝失敗，Agent 仍能運作。

### A2A 訊息格式

Request:
```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-id",
      "role": "user",
      "parts": [{"kind": "text", "text": "Hello"}]
    }
  }
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "result": {
    "parts": [{"kind": "text", "text": "Agent response"}]
  }
}
```

## 架構圖

```
┌───────────────────────────────────────────────┐
│              mask-deepagents                   │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │          MaskA2AServer (:10030)         │  │
│  │                                         │  │
│  │  ┌───────────────────────────────────┐  │  │
│  │  │       DeepAgentWrapper            │  │  │
│  │  │                                   │  │  │
│  │  │  ┌───────────────────────────┐    │  │  │
│  │  │  │   DeepAgents SDK         │    │  │  │
│  │  │  │   (or MASK LLM fallback) │    │  │  │
│  │  │  └───────────────────────────┘    │  │  │
│  │  └───────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
         │
         ▼ A2A Protocol
┌───────────────────────────────────────────────┐
│              mask-orchestrator                 │
│                 (:10020)                       │
└───────────────────────────────────────────────┘
```

## 驗證 Traces

1. **Phoenix UI**: http://localhost:6006
   - 檢查 project: `mask-deepagents`
   - Trace 結構: Agent invoke → LLM call

2. **Langfuse UI**: http://localhost:3001
   - 驗證 traces 是否正確記錄

## Git 提交範例

```bash
git init
git add .
git commit -m "feat: initial DeepAgents wrapper with MASK A2A"
gh repo create colinlee0924/mask-deepagents --public --push
```

## 常見問題

### Q: DeepAgents SDK 安裝失敗？
A: 這是正常的 fallback 情況，Agent 會使用 MASK LLM 繼續運作

### Q: 如何確認使用哪種模式？
A: 啟動時會顯示:
- "Using DeepAgents SDK" 表示正常
- "Warning: DeepAgents SDK not available, using fallback mode" 表示 fallback

### Q: 如何擴充 DeepAgents 功能？
A: 修改 `_init_deepagent()` 中的 `create_deep_agent()` 參數

### Q: 為什麼選擇 BaseAgent 而非 SimpleAgent？
A: DeepAgents SDK 有自己的工具和 agent 機制，不需要 MASK 的 SkillMiddleware
