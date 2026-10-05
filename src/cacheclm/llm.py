"""OpenAI-compatible chat client with a disk cache (replays are free), retries, and billed usage incl. cache-hit tokens."""
import hashlib
import json
import time
from pathlib import Path

import openai

from cacheclm.files import write_atomic

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)


def cost(reply, price):
    """USD of one call: cache hits at cache_read, other prompt tokens at cache_write, output at output."""
    hit = reply["cache_hit_tokens"]
    miss = reply["prompt_tokens"] - hit
    return (hit * price["cache_read"] + miss * price["cache_write"] + reply["completion_tokens"] * price["output"]) / 1e6


def _hits(usage):
    """Prompt tokens served from the provider's prefix cache (DeepSeek or OpenAI-style field)."""
    hits = getattr(usage, "prompt_cache_hit_tokens", None)
    if hits is None:
        details = getattr(usage, "prompt_tokens_details", None)
        hits = getattr(details, "cached_tokens", 0) if details else 0
    return hits or 0


class LLM:
    def __init__(self, model, client, cache_dir, temperature, price, options=None, on_spend=None, max_tries=4):
        self.model, self.client, self.cache_dir = model, client, Path(cache_dir)
        self.temperature, self.price, self.options = temperature, price, options or {}
        self.on_spend, self.max_tries = on_spend, max_tries

    def chat(self, messages, max_tokens, repeat):
        """{content, model, prompt_tokens, cache_hit_tokens, completion_tokens, latency_s, cached}."""
        payload = [self.model, self.options, self.temperature, repeat, max_tokens, messages]
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        reply = self._call(messages, max_tokens)
        write_atomic(path, json.dumps(reply))
        if self.on_spend:
            self.on_spend(cost(reply, self.price))
        return {**reply, "cached": False}

    def _call(self, messages, max_tokens):
        last = None
        for attempt in range(self.max_tries):
            start = time.monotonic()
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           temperature=self.temperature, max_tokens=max_tokens,
                                                           **self.options)
            except RETRYABLE as e:
                last = e
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            except openai.APIStatusError as e:
                if e.status_code in (401, 402):  # bad key or no balance: stop the whole run
                    raise
                raise RuntimeError(f"{self.model}: request rejected ({e.status_code}): {e}") from e
            return {"content": resp.choices[0].message.content or "", "model": resp.model,
                    "prompt_tokens": resp.usage.prompt_tokens, "cache_hit_tokens": _hits(resp.usage),
                    "completion_tokens": resp.usage.completion_tokens,
                    "latency_s": round(time.monotonic() - start, 3)}
        raise RuntimeError(f"{self.model}: gave up after {self.max_tries} tries: {last}")
