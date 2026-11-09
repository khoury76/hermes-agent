"""MiniMax's global Anthropic endpoint speaks the NATIVE adaptive-thinking contract.

MiniMax accepts Anthropic's ``thinking.budget_tokens`` but IGNORES it: budgets of
8k, 32k and 64k all produced identical reasoning, so every Hermes effort level
silently reasoned the same. The endpoint's real knob is
``output_config.effort`` (low/medium/high/xhigh/max) alongside
``thinking.type="adaptive"`` — verified live against
``https://api.minimax.io/anthropic/v1/messages``.

Sibling contracts:
  * ``thinking.type="disabled"`` 400s (the model requires adaptive thinking), and
    omitting ``thinking`` entirely still returns thinking blocks — MiniMax cannot
    think NOT at all, so "off" stays an omission there.
  * ``effort="ultra"`` 400s (``invalid reasoning_effort``); "ultra" is a Hermes-only
    label that clamps to "max" on the wire, exactly as it does for Anthropic.
  * The China endpoint (api.minimaxi.com) is NOT in scope: the global credential
    401s there, so its contract could not be verified and guessing would risk a 400
    on every turn.
"""

from __future__ import annotations

from agent.anthropic_adapter import build_anthropic_kwargs

MESSAGES = [{"role": "user", "content": "hello"}]
MINIMAX = "https://api.minimax.io/anthropic"
MINIMAX_CN = "https://api.minimaxi.com/anthropic"


def _kwargs(reasoning_config: dict, base_url: str = MINIMAX, model: str = "MiniMax-M3.1-Flash"):
    return build_anthropic_kwargs(
        model=model,
        messages=MESSAGES,
        tools=None,
        max_tokens=16384,
        reasoning_config=reasoning_config,
        base_url=base_url,
    )


def test_minimax_uses_native_adaptive_thinking_with_effort():
    kwargs = _kwargs({"enabled": True, "effort": "max"})
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "max"}
    # The ignored budget path must not linger alongside the native one.
    assert "budget_tokens" not in kwargs["thinking"]
    assert "temperature" not in kwargs


def test_minimax_effort_levels_map_to_native_vocabulary():
    for effort, expected in (
        ("low", "low"), ("medium", "medium"), ("high", "high"),
        ("xhigh", "xhigh"), ("max", "max"),
    ):
        kwargs = _kwargs({"enabled": True, "effort": effort})
        assert kwargs["output_config"] == {"effort": expected}, effort


def test_minimax_ultra_clamps_to_max():
    """Hermes' "ultra" is a label, not a MiniMax level: it must never reach the wire."""
    kwargs = _kwargs({"enabled": True, "effort": "ultra"})
    assert kwargs["output_config"] == {"effort": "max"}
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_minimax_minimal_clamps_to_low():
    kwargs = _kwargs({"enabled": True, "effort": "minimal"})
    assert kwargs["output_config"] == {"effort": "low"}


def test_minimax_disabled_reasoning_stays_an_omission():
    """MiniMax 400s on thinking.type=disabled and cannot think NOT at all."""
    kwargs = _kwargs({"enabled": False})
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


def test_minimax_cn_endpoint_keeps_the_legacy_contract():
    """Unverified endpoint: the global credential 401s there, so it must not be assumed."""
    kwargs = _kwargs({"enabled": True, "effort": "max"}, base_url=MINIMAX_CN)
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_other_anthropic_endpoints_are_unaffected():
    """Claude keeps its own adaptive contract; unrelated hosts keep the legacy one."""
    claude = build_anthropic_kwargs(
        model="claude-opus-4-7", messages=MESSAGES, tools=None, max_tokens=16384,
        reasoning_config={"enabled": True, "effort": "max"}, base_url="https://api.anthropic.com",
    )
    assert claude["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert claude["output_config"] == {"effort": "max"}

    other = build_anthropic_kwargs(
        model="some-model", messages=MESSAGES, tools=None, max_tokens=16384,
        reasoning_config={"enabled": True, "effort": "high"}, base_url="https://example.com",
    )
    assert other["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert "output_config" not in other
