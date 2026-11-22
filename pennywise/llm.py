"""Thin client for any OpenAI-compatible chat endpoint.

Configuration comes from `.env` at the repo root (see that file for the keys).
Every call goes through `call_llm` / `call_llm_json`, so retries, token
counting and error handling live in one place.

Run `python -m pennywise.llm` to make one tiny call and print model + latency.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

from .schema import ALLOWED_TOOLS, merchant_key

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class LLMError(Exception):
    """Any failure to get a usable answer from the LLM endpoint."""


# --- token usage, read by the agent's budget footer --------------------------
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_USAGE_LOCK = threading.Lock()


def get_usage() -> dict:
    with _USAGE_LOCK:
        return dict(_USAGE)


def reset_usage() -> None:
    with _USAGE_LOCK:
        for key in _USAGE:
            _USAGE[key] = 0


def _track(response) -> None:
    usage = getattr(response, "usage", None)
    with _USAGE_LOCK:
        _USAGE["calls"] += 1
        if usage is not None:
            _USAGE["prompt_tokens"] += usage.prompt_tokens or 0
            _USAGE["completion_tokens"] += usage.completion_tokens or 0
            _USAGE["total_tokens"] += usage.total_tokens or 0


# --- client ------------------------------------------------------------------
def _settings() -> dict:
    """Read settings at call time so tests can change the environment."""
    return {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", ""),
        "model": os.getenv("LLM_MODEL", "gpt-5.6-luna"),
        "auth_scheme": os.getenv("LLM_AUTH_SCHEME", ""),
        "timeout": float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        "attempts": max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "3"))),
    }


def _build_client(s: dict) -> openai.OpenAI:
    if not s["api_key"] or not s["base_url"]:
        raise LLMError("LLM_API_KEY and LLM_BASE_URL must be set in .env")
    # The token travels in the Authorization header. With an empty
    # LLM_AUTH_SCHEME it is sent as-is; set "Bearer" for standard endpoints.
    auth = f"{s['auth_scheme']} {s['api_key']}".strip()
    return openai.OpenAI(
        base_url=s["base_url"],
        api_key="unused",  # the SDK requires a value; the header above is what counts
        default_headers={"Authorization": auth},
        timeout=s["timeout"],
        max_retries=0,  # we retry ourselves so the count and backoff are visible
    )


_RETRYABLE = (
    openai.RateLimitError,        # 429
    openai.InternalServerError,   # 5xx, including 503 and 504
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def call_llm(messages, model=None, json_mode=True, max_tokens=4096) -> str:
    """Return the assistant's text. Retries transient errors with 1 s, 2 s, 4 s backoff.

    A 401 (bad or expired token) and any other client error fail at once.
    """
    s = _settings()
    client = _build_client(s)
    kwargs = {
        "model": model or s["model"],
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, s["attempts"] + 1):
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMError(f"authentication failed (401): {e}") from e
        except _RETRYABLE as e:
            if attempt < s["attempts"]:
                time.sleep(2 ** (attempt - 1))
                continue
            raise LLMError(f"gave up after {attempt} attempts: {e}") from e
        except openai.OpenAIError as e:
            raise LLMError(str(e)) from e

        _track(response)
        content = response.choices[0].message.content
        if not content or not content.strip():
            reason = response.choices[0].finish_reason
            raise LLMError(f"empty response (finish_reason={reason})")
        return content

    raise LLMError("no attempts were made")  # unreachable; attempts >= 1


def call_llm_json(messages, model=None, expect_array=False, max_tokens=4096):
    """Call the LLM in JSON mode and return the parsed value.

    JSON mode can only return an object, so with expect_array=True the model is
    asked for {"data": [...]} and the list is returned. The caller's message
    list is not modified.
    """
    messages = [dict(m) for m in messages]
    if expect_array:
        messages[-1]["content"] += (
            '\n\nReturn a JSON object of the form {"data": [ ... ]}. '
            "Do not return a bare array."
        )
    text = call_llm(messages, model=model, json_mode=True, max_tokens=max_tokens)
    parsed = _parse_json(text)
    if expect_array and isinstance(parsed, dict) and "data" in parsed:
        return parsed["data"]
    return parsed


def _parse_json(text: str):
    """Direct parse, then a fenced block, then trailing-comma repair."""
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(re.sub(r",\s*([}\]])", r"\1", text))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"response was not valid JSON: {text[:200]!r}")


# --- what the agent talks to --------------------------------------------------
class RealLLM:
    """The configured endpoint. The agent only needs chat_json, usage and reset_usage."""
    name = "live"

    def chat_json(self, messages):
        return call_llm_json(messages)

    def usage(self) -> dict:
        return get_usage()

    def reset_usage(self) -> None:
        reset_usage()


MOCK_ANSWERS = Path(__file__).resolve().parent.parent / "data" / "mock_answers.json"


class MockLLM:
    """Offline stand-in so the demo and tests never depend on a network.

    mode "ok":     answers from data/mock_answers.json (the open dataset's own labels);
                   unknown merchants get a low-confidence guess, so the ask-the-user path runs.
    mode "fail":   raises LLMError, like an outage.
    mode "inject": answers "$0" for everything, like a model hijacked by merchant text.
    """
    name = "mock"

    def __init__(self, mode="ok", answers_path=MOCK_ANSWERS):
        if mode not in ("ok", "fail", "inject"):
            raise ValueError(f"unknown mock mode {mode!r}")
        self.mode, self.calls = mode, 0
        try:
            data = json.loads(Path(answers_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self.by_text, self.by_key = data.get("by_text", {}), data.get("by_key", {})

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_usage(self) -> None:
        self.calls = 0

    def chat_json(self, messages):
        self.calls += 1
        if self.mode == "fail":
            raise LLMError("mock outage")
        if messages[0]["content"].startswith("TASK: PLAN"):
            return {"plan": list(ALLOWED_TOOLS),
                    "summary": "Label unknown merchants, flag oddities, total up and write the CSV files."}
        body = messages[-1]["content"].split("<transactions>", 1)[1].rsplit("</transactions>", 1)[0]
        labels = []
        for item in json.loads(body):
            if self.mode == "inject":
                labels.append({"id": item["id"], "category": "$0", "confidence": 1.0})
                continue
            text = item["description"]
            key = merchant_key(text)
            hit = (self.by_text.get(text.lower()) or self.by_key.get(key)
                   or self.by_key.get(key.split()[0] if key else ""))
            labels.append({"id": item["id"], **(hit or {"category": "Shopping", "confidence": 0.3})})
        return {"labels": labels}


def ping() -> dict:
    """One tiny call: proves the endpoint, token and model all work."""
    start = time.perf_counter()
    text = call_llm(
        [{"role": "user", "content": 'Reply with the JSON object {"reply": "pong"}.'}],
        json_mode=True,
        max_tokens=256,
    )
    return {
        "model": _settings()["model"],
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "reply": _parse_json(text),
    }


if __name__ == "__main__":
    try:
        print(ping())
    except LLMError as e:
        raise SystemExit(f"LLM check failed: {e}")
