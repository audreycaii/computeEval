"""List prices, USD per 1M tokens (input, output), October 2026."""
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, usage) -> float:
    pin, pout = PRICES[model]
    return usage.input_tokens * pin / 1e6 + usage.output_tokens * pout / 1e6
