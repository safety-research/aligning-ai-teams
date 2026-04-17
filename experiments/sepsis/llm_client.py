#!/usr/bin/env python3
"""
LLM client using the Anthropic SDK directly.

Drop-in replacement for the previous safety-tooling based client.
Only dependency: anthropic (+ tiktoken for token counting).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
import tiktoken

DEFAULT_CONFIG = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 20000,
    "temperature": 0.1,
}
# Initialize conversation truncation based on max_tokens
# Use 80% for conversation, leave 20% for response generation
CONVERSATION_PERCENTAGE = 0.80
MIN_CONVERSATION_TOKENS = 2000  # Minimum floor


class LLMClient:
    """LLM client using the Anthropic SDK with automatic conversation truncation."""

    def __init__(
        self,
        config_dict: Dict[str, Any] = None,
        cache_dir: Path = None,
        suppress_warnings: bool = True,
        use_cache: bool = True,
    ):
        self.config = config_dict or DEFAULT_CONFIG
        self.model_id = self.config.get("model")
        self.temperature = self.config.get("temperature")
        self.max_tokens = self.config.get("max_tokens")
        self.logger = logging.getLogger(__name__)

        self.client = anthropic.AsyncAnthropic(max_retries=8)  # reads ANTHROPIC_API_KEY from env

        calculated_max = int(self._get_context_window_for_model(self.model_id) * CONVERSATION_PERCENTAGE)
        self.conversation_max_tokens = max(calculated_max, MIN_CONVERSATION_TOKENS)

        # Initialize encoder based on model
        self.encoder = self._get_encoder_for_model(self.model_id)

    def _get_encoder_for_model(self, model_id: str) -> Optional[object]:
        if not model_id:
            return None
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Failed to load tiktoken encoder for model '{model_id}': {e}")
            return None

    def _get_context_window_for_model(self, _model_id: str) -> int:
        return 200000

    def _count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4

    def _count_message_tokens(self, message: Dict[str, str]) -> int:
        role_tokens = self._count_tokens(message.get("role", ""))
        content_tokens = self._count_tokens(message.get("content", ""))
        return role_tokens + content_tokens + 4

    def _count_conversation_tokens(self, messages: List[Dict[str, str]], system_message: Optional[str] = None) -> int:
        total = 0
        if system_message:
            total += self._count_tokens(system_message) + 4
        for message in messages:
            total += self._count_message_tokens(message)
        return total

    def _truncate_conversation_if_needed(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        keep_first_n: int = 2,
        keep_last_n: int = 15,
        summarize_removed: bool = True,
    ) -> Tuple[List[Dict[str, str]], bool]:
        current_tokens = self._count_conversation_tokens(messages, system_message)

        if current_tokens <= self.conversation_max_tokens:
            return messages.copy(), False

        self.logger.warning(
            f"Conversation exceeds token limit ({current_tokens} > {self.conversation_max_tokens}). "
            f"Truncating from {len(messages)} messages..."
        )

        truncated = []

        if keep_first_n > 0 and len(messages) > keep_first_n:
            truncated.extend(messages[:keep_first_n])

        if summarize_removed and len(messages) > keep_first_n + keep_last_n:
            removed_count = len(messages) - keep_first_n - keep_last_n
            truncation_notice = {
                "role": "user",
                "content": f"[Note: {removed_count} intermediate messages were removed to stay within token limits. The conversation continues from recent context.]",
            }
            truncated.append(truncation_notice)

        if keep_last_n > 0 and len(messages) > keep_last_n:
            truncated.extend(messages[-keep_last_n:])
        else:
            truncated = messages.copy()

        while self._count_conversation_tokens(truncated, system_message) > self.conversation_max_tokens:
            if len(truncated) <= keep_first_n + 2:
                break
            middle_idx = keep_first_n + 1
            if middle_idx < len(truncated) - 2:
                truncated.pop(middle_idx)
            else:
                truncated.pop(-3)

        new_tokens = self._count_conversation_tokens(truncated, system_message)
        self.logger.info(f"Truncated to {len(truncated)} messages ({new_tokens} tokens)")

        return truncated, True

    async def generate_response(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            model_id = self.model_id
            temperature = self.temperature
            max_tokens = self.max_tokens

            if config_override:
                model_id = config_override.get("model", model_id)
                temperature = config_override.get("temperature", temperature)
                max_tokens = config_override.get("max_tokens", max_tokens)

            # Build messages for the Anthropic API
            api_messages = []

            if isinstance(prompt, str):
                api_messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                safe_conversation, _ = self._truncate_conversation_if_needed(
                    prompt, system_message, keep_first_n=2, keep_last_n=15, summarize_removed=True
                )
                for msg in safe_conversation:
                    if msg["role"] in ("user", "assistant"):
                        api_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                raise ValueError(f"Unsupported prompt type: {type(prompt)}")

            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages,
            }
            if system_message:
                kwargs["system"] = system_message

            response = await self.client.messages.create(**kwargs)

            # Extract text from response
            for block in response.content:
                if block.type == "text":
                    return block.text

            return ""

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    @classmethod
    def from_config_dict(
        cls, config_dict: Dict[str, Any], suppress_warnings: bool = True, use_cache: bool = True
    ) -> "LLMClient":
        return cls(config_dict=config_dict, suppress_warnings=suppress_warnings, use_cache=use_cache)


# Backwards-compatible aliases
SafetyToolingLLMClient = LLMClient


def create_llm_client(
    config_dict: Dict[str, Any] = None, suppress_warnings: bool = True, use_cache: bool = True
) -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient.from_config_dict(config_dict, suppress_warnings=suppress_warnings, use_cache=use_cache)
