import backoff  # for exponential backoff
import openai
import json
import asyncio
from time import perf_counter
import threading
import time
from collections import deque
from typing import Any
from retrying import retry
from stock_disagreement.config import (
    LLM_INPUT_COST_PER_MILLION,
    LLM_OUTPUT_COST_PER_MILLION,
    LLM_REQUESTS_PER_MINUTE,
    LLM_CACHE_DIR,
    LLM_CACHE_MODE,
)
from stock_disagreement.llm_cache import LLMResponseCache


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = max(int(requests_per_minute), 0)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.requests_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                wait_seconds = max(60.0 - (now - self._timestamps[0]), 0.01)
            time.sleep(wait_seconds)


GLOBAL_REQUEST_LIMITER = RequestRateLimiter(LLM_REQUESTS_PER_MINUTE)
GLOBAL_RESPONSE_CACHE = LLMResponseCache(LLM_CACHE_DIR, LLM_CACHE_MODE)

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def completions_with_backoff(client, **kwargs):
    GLOBAL_REQUEST_LIMITER.acquire()
    return client.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    client,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = []
    for messages in messages_list:
        GLOBAL_REQUEST_LIMITER.acquire()
        async_responses.append(client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        ))
    return await asyncio.gather(*async_responses)


class OpenAIModel:
    def __init__(self, model_name, stop_words, max_new_tokens) -> None:
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens
        self.call_metrics: list[dict[str, Any]] = []
        self._metrics_lock = threading.Lock()
        self.last_call_metrics: dict[str, Any] | None = None

    @retry(stop_max_attempt_number=10, wait_fixed=2000)
    def chat_generate(self, client, system_prompt:str, input_string:str, temperature = 0.0):
        cache_payload = {
            "cache_schema": 2,
            "model": self.model_name,
            "system_prompt": system_prompt,
            "input": input_string,
            "temperature": temperature,
            "max_new_tokens": self.max_new_tokens,
            "stop_words": self.stop_words,
        }
        cache_key = GLOBAL_RESPONSE_CACHE.key(cache_payload)
        cached = GLOBAL_RESPONSE_CACHE.get(cache_key)
        if cached is None and GLOBAL_RESPONSE_CACHE.mode == "read":
            cached = GLOBAL_RESPONSE_CACHE.require_hit(cache_key)
        if cached is not None:
            historical_metrics = dict(cached.get("metrics", {}))
            historical_metrics.update({
                "model": self.model_name,
                "success": True,
                "finish_reason": cached.get("finish_reason", "cached"),
                "cached": True,
                "cache_key": cache_key,
                "billed_prompt_tokens": 0,
                "billed_completion_tokens": 0,
                "billed_total_tokens": 0,
                "billed_latency_seconds": 0.0,
                "billed_monetary_cost": 0.0,
            })
            self._record_metrics(historical_metrics)
            return str(cached["text"]), str(cached.get("finish_reason", "cached"))
        started = perf_counter()
        try:
            response = completions_with_backoff(
                client,
                model = self.model_name,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_string}
                    ],
                temperature = temperature,
                top_p = 1,
                stop = self.stop_words
            )
        except Exception as exc:
            self._record_metrics({
                "model": self.model_name,
                "latency_seconds": perf_counter() - started,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "success": False,
                "error_type": type(exc).__name__,
            })
            raise
        generated_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        metrics = {
            "model": self.model_name,
            "latency_seconds": perf_counter() - started,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "monetary_cost": (
                prompt_tokens * LLM_INPUT_COST_PER_MILLION
                + completion_tokens * LLM_OUTPUT_COST_PER_MILLION
            ) / 1_000_000.0,
            "success": True,
            "finish_reason": finish_reason,
            "cached": False,
            "cache_key": cache_key,
        }
        metrics.update({
            "billed_prompt_tokens": metrics["prompt_tokens"],
            "billed_completion_tokens": metrics["completion_tokens"],
            "billed_total_tokens": metrics["total_tokens"],
            "billed_latency_seconds": metrics["latency_seconds"],
            "billed_monetary_cost": metrics["monetary_cost"],
        })
        self._record_metrics(metrics)
        GLOBAL_RESPONSE_CACHE.put(cache_key, {
            "text": generated_text,
            "finish_reason": finish_reason,
            "metrics": metrics,
        })
        return generated_text, finish_reason

    def _record_metrics(self, metrics: dict[str, Any]) -> None:
        with self._metrics_lock:
            self.last_call_metrics = metrics.copy()
            self.call_metrics.append(metrics.copy())

    def generate(self, client, input_string, temperature = 0.0):
        return self.chat_generate(client, input_string, temperature)
    
    def batch_chat_generate(self, client, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    client, open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        finish_reason = [x.choices[0].finish_reason.strip() for x in predictions]
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, client, messages_list, temperature = 0.0):
        return self.batch_chat_generate(client, messages_list, temperature)
