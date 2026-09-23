"""
Spatial Atlas: Unified LLM Interface

Wraps litellm for multi-provider model access with cost tracking.
Supports text generation, JSON mode, and vision analysis.
"""

import base64
import logging

import litellm

from config import Config
from cost.tracker import CostTracker

logger = logging.getLogger("spatial-atlas.llm")

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.cost_tracker = CostTracker()

    def _get_model(self, model_tier: str) -> str:
        return self.config.model_tiers.get(model_tier, self.config.standard_model)

    def _apply_chat_template_options(self, kwargs: dict) -> None:
        """Apply optional vLLM chat-template controls to one request."""
        enable_thinking = self.config.llm_enable_thinking
        if enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    async def generate(
        self,
        prompt: str,
        *,
        model_tier: str = "standard",
        system_prompt: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text completion using specified model tier."""
        model = self._get_model(model_tier)

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self._apply_chat_template_options(kwargs)

        try:
            response = await litellm.acompletion(**kwargs)
            self.cost_tracker.track(
                response,
                role=model_tier,
                model_tier=model_tier,
                call_kind="generate",
                configured_max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM [{model_tier}] generated {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM generation failed [{model_tier}]: {e}")
            raise

    async def vision_analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        model_tier: str = "vision",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Analyze an image using a multimodal model."""
        model = self._get_model(model_tier)
        b64 = base64.b64encode(image_bytes).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ]

        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            self._apply_chat_template_options(kwargs)
            response = await litellm.acompletion(**kwargs)
            self.cost_tracker.track(
                response,
                role=model_tier,
                model_tier=model_tier,
                call_kind="vision_analyze",
                configured_max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"Vision analysis generated {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise

    async def generate_with_messages(
        self,
        messages: list[dict],
        *,
        model_tier: str = "standard",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate with full message control (for multi-turn)."""
        model = self._get_model(model_tier)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self._apply_chat_template_options(kwargs)

        try:
            response = await litellm.acompletion(**kwargs)
            self.cost_tracker.track(
                response,
                role=model_tier,
                model_tier=model_tier,
                call_kind="generate_with_messages",
                configured_max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
