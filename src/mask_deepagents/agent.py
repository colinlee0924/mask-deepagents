"""DeepAgents SDK wrapper with MASK interface.

This module wraps the DeepAgents SDK to provide a MASK-compatible interface
that can be exposed via A2A protocol.
"""

import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

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
            model_name: Model to use (defaults to env DEEPAGENT_MODEL or claude-sonnet).
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
            handoff_context: Optional handoff context from parent agent.

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
