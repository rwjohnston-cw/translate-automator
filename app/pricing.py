from __future__ import annotations

from dataclasses import dataclass

# Pricing sources checked June 2026:
# - OpenAI: https://developers.openai.com/api/docs/pricing
# - Gemini: https://ai.google.dev/gemini-api/docs/pricing
# - Gemini 3 table: https://ai.google.dev/gemini-api/docs/gemini-3
# - DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
# - Gemma 4 31B IT fallback market rate (no single official public rate in Gemini docs):
#   https://modelcompare.dev/providers/google


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_usd: float
    output_per_million_usd: float
    source: str
    notes: str | None = None


MODEL_PRICING_USD: dict[str, dict[str, ModelPricing]] = {
    "openai": {
        "gpt-5.5": ModelPricing(5.00, 30.00, "https://developers.openai.com/api/docs/pricing"),
        "gpt-5.4": ModelPricing(2.50, 15.00, "https://developers.openai.com/api/docs/pricing"),
        "gpt-5.4-mini": ModelPricing(
            0.75,
            4.50,
            "https://developers.openai.com/api/docs/pricing",
        ),
        "gpt-5.4-nano": ModelPricing(
            0.20,
            1.25,
            "https://developers.openai.com/api/docs/pricing",
        ),
        "gpt-5-mini": ModelPricing(
            0.25,
            2.00,
            "https://developers.openai.com/api/docs/pricing",
        ),
        "gpt-5-nano": ModelPricing(
            0.05,
            0.40,
            "https://developers.openai.com/api/docs/pricing",
        ),
    },
    "gemini": {
        "gemini-2.5-flash-lite": ModelPricing(
            0.10,
            0.40,
            "https://ai.google.dev/gemini-api/docs/pricing",
        ),
        "gemini-2.5-flash": ModelPricing(
            0.30,
            2.50,
            "https://ai.google.dev/gemini-api/docs/pricing",
        ),
        "gemini-2.5-pro": ModelPricing(
            1.25,
            10.00,
            "https://ai.google.dev/gemini-api/docs/pricing",
            notes="Uses <=200k input token tier for estimate.",
        ),
        "gemini-3-flash-preview": ModelPricing(
            0.50,
            3.00,
            "https://ai.google.dev/gemini-api/docs/gemini-3",
        ),
        "gemini-3.1-flash-lite": ModelPricing(
            0.25,
            1.50,
            "https://ai.google.dev/gemini-api/docs/gemini-3",
        ),
        "gemma-4-31b-it": ModelPricing(
            0.10,
            0.30,
            "https://modelcompare.dev/providers/google",
            notes="Ecosystem market rate; provider-specific billing may vary.",
        ),
    },
    "deepseek": {
        "deepseek-v4-flash": ModelPricing(
            0.14,
            0.28,
            "https://api-docs.deepseek.com/quick_start/pricing",
        ),
        "deepseek-v4-pro": ModelPricing(
            0.435,
            0.87,
            "https://api-docs.deepseek.com/quick_start/pricing",
        ),
    },
}


def estimate_cost_usd(
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> dict[str, float | str | None]:
    provider_rates = MODEL_PRICING_USD.get(provider, {})
    pricing = provider_rates.get(model)
    if pricing is None:
        return {
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "pricing_source": None,
            "pricing_notes": "No pricing entry found for model.",
        }

    in_tokens = max(0, input_tokens or 0)
    out_tokens = max(0, output_tokens or 0)
    input_cost = (in_tokens / 1_000_000) * pricing.input_per_million_usd
    output_cost = (out_tokens / 1_000_000) * pricing.output_per_million_usd
    total_cost = input_cost + output_cost
    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
        "pricing_source": pricing.source,
        "pricing_notes": pricing.notes,
    }

