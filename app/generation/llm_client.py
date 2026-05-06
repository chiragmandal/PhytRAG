"""
LLM client using the OpenAI Python library pointed at Ollama's
OpenAI-compatible endpoint (http://host:11434/v1).

Why the OpenAI library instead of raw httpx?
  1. It handles streaming protocol details correctly.
  2. When the team upgrades from Ollama to vLLM or any other OpenAI-compatible
     server, this code changes ONLY the base_url - nothing else.
  3. The retry/backoff logic is built-in.

The client is instantiated per-request (stateless). If this becomes a
bottleneck, pool AsyncOpenAI instances in app.state.
"""
import logging
import time

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Ollama's OpenAI-compatible endpoint
        self._client = AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",  # value is required but not validated by Ollama
        )

    async def generate(self, prompt: str) -> tuple[str, int, int]:
        """
        Stream the response so we can capture time-to-first-token (TTFT).
        Returns (answer_text, ttft_ms, tokens_generated).

        TTFT is the primary latency metric users feel in interactive applications.
        We record it separately from total latency in Prometheus.
        """
        t_start = time.perf_counter()
        ttft_ms = 0
        chunks: list[str] = []
        tokens_generated = 0
        first_token_seen = False

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if not first_token_seen:
                        ttft_ms = int((time.perf_counter() - t_start) * 1000)
                        first_token_seen = True
                    chunks.append(delta)
                    tokens_generated += 1  # approximate: one chunk ~ one token

            answer = "".join(chunks).strip()
            logger.debug(
                "LLM generation complete: model=%s tokens=%d ttft_ms=%d",
                self.model,
                tokens_generated,
                ttft_ms,
            )
            return answer, ttft_ms, tokens_generated

        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            raise
