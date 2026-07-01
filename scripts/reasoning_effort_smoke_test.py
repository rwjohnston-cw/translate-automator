from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from openai import OpenAI

# Allow running this file directly from repo root:
#   ./.venv/bin/python scripts/reasoning_effort_smoke_test.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings
from app.models import LLM_PROVIDER_DEEPSEEK, LLM_PROVIDER_GEMINI, LLM_PROVIDER_OPENAI

SHOW_COLOR = sys.stdout.isatty()
PROGRESS_BAR_WIDTH = 28

# Keep this in sync with app/main.py.
MODEL_OPTIONS: dict[str, list[str]] = {
    LLM_PROVIDER_OPENAI: [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    LLM_PROVIDER_GEMINI: [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
    ],
    LLM_PROVIDER_DEEPSEEK: [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ],
}

# Keep this in sync with app/main.py.
REASONING_OPTIONS: dict[str, dict[str, list[str]]] = {
    LLM_PROVIDER_OPENAI: {
        "__default__": ["none", "minimal", "low", "medium", "high", "xhigh"],
        "gpt-5.5": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4-mini": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4-nano": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5-mini": ["minimal", "low", "medium", "high"],
        "gpt-5-nano": ["minimal", "low", "medium", "high"],
    },
    LLM_PROVIDER_GEMINI: {
        "__default__": ["low", "medium", "high"],
        "gemini-2.5-flash-lite": ["low", "medium", "high"],
        "gemini-2.5-flash": ["low", "medium", "high"],
        "gemini-2.5-pro": ["low", "medium", "high"],
        "gemini-3-flash-preview": ["low", "medium", "high"],
        "gemini-3.1-flash-lite": ["low", "medium", "high"],
        "gemma-4-31b-it": ["high"],
    },
    LLM_PROVIDER_DEEPSEEK: {
        "__default__": ["high", "max"],
        "deepseek-v4-flash": ["high", "max"],
        "deepseek-v4-pro": ["high", "max"],
    },
}

GEMINI_BUDGET_BY_EFFORT = {
    "low": 1024,
    "medium": 8192,
    "high": 16384,
}


@dataclass
class SmokeResult:
    provider: str
    model: str
    reasoning_effort: str
    outcome: Literal["PASS", "FAIL", "SKIP"]
    duration_ms: int
    details: str


def _color(text: str, code: str) -> str:
    if not SHOW_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _outcome_badge(outcome: Literal["PASS", "FAIL", "SKIP"]) -> str:
    if outcome == "PASS":
        return _color("PASS", "32")
    if outcome == "FAIL":
        return _color("FAIL", "31")
    return _color("SKIP", "33")


def _render_progress(current: int, total: int) -> str:
    if total <= 0:
        return "[............................] 0/0"
    filled = int((current / total) * PROGRESS_BAR_WIDTH)
    filled = max(0, min(PROGRESS_BAR_WIDTH, filled))
    bar = "#" * filled + "." * (PROGRESS_BAR_WIDTH - filled)
    percent = int((current / total) * 100)
    return f"[{bar}] {current}/{total} ({percent:>3}%)"


def _supported_reasoning(provider: str, model: str) -> list[str]:
    provider_options = REASONING_OPTIONS.get(provider, {})
    return provider_options.get(model) or provider_options.get("__default__", [])


def _select_reasoning_effort(provider: str, model: str) -> str:
    supported = _supported_reasoning(provider, model)
    if not supported:
        return ""
    if provider == LLM_PROVIDER_DEEPSEEK:
        if model.endswith("-pro") and "max" in supported:
            return "max"
        if "high" in supported:
            return "high"
    if "low" in supported:
        return "low"
    if "minimal" in supported:
        return "minimal"
    return supported[0]


def _build_openai_client(api_key: str, base_url: str | None = None) -> OpenAI:
    kwargs = {"api_key": api_key, "timeout": 60}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _run_openai_call(client: OpenAI, model: str, reasoning_effort: str) -> str:
    response = client.responses.create(
        model=model,
        input="Reply with exactly OK.",
        reasoning={"effort": reasoning_effort},
        max_output_tokens=32,
    )
    return (response.output_text or "").strip() or "<empty>"


def _run_gemini_call(api_key: str, model: str, reasoning_effort: str) -> str:
    thinking_config: dict[str, int | str]
    if model.startswith("gemini-2.5-"):
        budget = GEMINI_BUDGET_BY_EFFORT.get(reasoning_effort)
        if budget is None:
            raise RuntimeError(f"Unsupported Gemini 2.5 effort: {reasoning_effort}")
        thinking_config = {"thinkingBudget": budget}
    else:
        thinking_config = {"thinkingLevel": reasoning_effort}

    payload = {
        "systemInstruction": {
            "parts": [{"text": "You are a concise assistant. Reply with exactly OK."}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Say OK"}],
            }
        ],
        "generationConfig": {
            "thinkingConfig": thinking_config,
            "maxOutputTokens": 32,
        },
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    with httpx.Client(timeout=60) as client:
        response = client.post(
            endpoint,
            headers={"x-goog-api-key": api_key},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    payload_json = response.json()
    candidates = payload_json.get("candidates") or []
    if not candidates:
        return "<empty_candidates>"
    for part in candidates[0].get("content", {}).get("parts", []):
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "<no_text_part>"


def _run_deepseek_call(client: OpenAI, model: str, reasoning_effort: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant. Reply with exactly OK."},
            {"role": "user", "content": "Say OK"},
        ],
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=64,
    )
    content = (response.choices[0].message.content or "").strip()
    return content or "<empty>"


def _run_for_model(settings: Settings, provider: str, model: str) -> SmokeResult:
    effort = _select_reasoning_effort(provider, model)
    if not effort:
        return SmokeResult(
            provider=provider,
            model=model,
            reasoning_effort="<none>",
            outcome="SKIP",
            duration_ms=0,
            details="No reasoning effort mapping found",
        )

    start = time.perf_counter()
    try:
        if provider == LLM_PROVIDER_OPENAI:
            if not settings.openai_api_key:
                return SmokeResult(provider, model, effort, "SKIP", 0, "OPENAI_API_KEY missing")
            client = _build_openai_client(settings.openai_api_key.get_secret_value())
            output = _run_openai_call(client, model, effort)
        elif provider == LLM_PROVIDER_GEMINI:
            if not settings.gemini_api_key:
                return SmokeResult(provider, model, effort, "SKIP", 0, "GEMINI_API_KEY missing")
            output = _run_gemini_call(settings.gemini_api_key.get_secret_value(), model, effort)
        elif provider == LLM_PROVIDER_DEEPSEEK:
            if not settings.deepseek_api_key:
                return SmokeResult(provider, model, effort, "SKIP", 0, "DEEPSEEK_API_KEY missing")
            client = _build_openai_client(
                settings.deepseek_api_key.get_secret_value(),
                settings.deepseek_base_url.rstrip("/"),
            )
            output = _run_deepseek_call(client, model, effort)
        else:
            return SmokeResult(provider, model, effort, "SKIP", 0, "Unsupported provider")
    except Exception as exc:  # pragma: no cover - integration smoke path
        duration_ms = int((time.perf_counter() - start) * 1000)
        return SmokeResult(provider, model, effort, "FAIL", duration_ms, str(exc))

    duration_ms = int((time.perf_counter() - start) * 1000)
    preview = output.replace("\n", "\\n")[:80]
    return SmokeResult(provider, model, effort, "PASS", duration_ms, f"response={preview}")


def _format_results(results: list[SmokeResult]) -> str:
    lines = []
    header = f"{'OUTCOME':<7}  {'PROVIDER':<8}  {'MODEL':<24}  {'EFFORT':<8}  {'MS':>6}  DETAILS"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        outcome_label = _outcome_badge(result.outcome)
        lines.append(
            f"{outcome_label:<7}  {result.provider:<8}  {result.model:<24}  "
            f"{result.reasoning_effort:<8}  {result.duration_ms:>6}  {result.details}"
        )
    totals = {
        "PASS": sum(1 for r in results if r.outcome == "PASS"),
        "FAIL": sum(1 for r in results if r.outcome == "FAIL"),
        "SKIP": sum(1 for r in results if r.outcome == "SKIP"),
    }
    lines.append("")
    lines.append(f"Totals: PASS={totals['PASS']} FAIL={totals['FAIL']} SKIP={totals['SKIP']}")
    return "\n".join(lines)


def main() -> int:
    settings = Settings()
    run_plan = [(provider, model) for provider, models in MODEL_OPTIONS.items() for model in models]
    total = len(run_plan)
    results: list[SmokeResult] = []
    suite_start = time.perf_counter()

    print(_color("Starting reasoning-effort smoke test", "1;36"))
    print(f"Models queued: {total}")
    print(f"Progress      {_render_progress(0, total)}")

    for idx, (provider, model) in enumerate(run_plan, start=1):
        effort = _select_reasoning_effort(provider, model) or "<none>"
        print(
            _color(
                f"\n[{idx:02d}/{total:02d}] Running {provider}:{model} (effort={effort}) ...",
                "36",
            ),
            flush=True,
        )
        result = _run_for_model(settings, provider, model)
        results.append(result)
        print(
            f" -> {_outcome_badge(result.outcome)} in {result.duration_ms}ms | {result.details}",
            flush=True,
        )
        print(f"Progress      {_render_progress(idx, total)}", flush=True)

    total_ms = int((time.perf_counter() - suite_start) * 1000)
    print(f"\nCompleted in {total_ms}ms\n")

    print(_format_results(results))
    failed = any(item.outcome == "FAIL" for item in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
