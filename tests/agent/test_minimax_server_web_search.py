"""Tests for the opt-in MiniMax server-side web_search (web_search_20250305) wiring.

Covers the approved plan in t_5e2107a1:
- outbound kwargs wiring in build_anthropic_kwargs (flag off/on, provider gates,
  tool_choice "none" and forced-tool cases)
- normalize_response handling of server_tool_use blocks (sources -> provider_data)
- streaming _call_anthropic handling (text intact, sources collected)
- interleaved replay round-trip (server_tool_use + client tool_use + signed thinking)
"""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs

_MINIMAX_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]

_SERVER_ENTRY = {"type": "web_search_20250305", "name": "web_search"}


def _config_on():
    return {"web": {"server_web_search": True}}


def _kwargs(tools=None, tool_choice=None, provider=None):
    return build_anthropic_kwargs(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        max_tokens=4096,
        reasoning_config=None,
        tool_choice=tool_choice,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Outbound kwargs wiring
# ---------------------------------------------------------------------------

class TestServerWebSearchKwargs:
    def test_flag_off_byte_identical(self):
        """Flag off => outbound kwargs contain NO server tool entry, even for minimax,
        and are identical to a build that never knew the provider."""
        cfg = {"web": {"server_web_search": False}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            with_provider = _kwargs(tools=_MINIMAX_TOOLS, provider="minimax")
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            legacy = _kwargs(tools=_MINIMAX_TOOLS)
        assert with_provider == legacy
        assert all(t.get("name") != "web_search" for t in with_provider.get("tools", []))

    def test_flag_off_default_config_no_key(self):
        """No key at all in config => off (None is not True)."""
        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            kwargs = _kwargs(tools=_MINIMAX_TOOLS, provider="minimax")
        assert all(t.get("name") != "web_search" for t in kwargs.get("tools", []))

    def test_flag_on_minimax_appends_after_function_tools(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
            kwargs = _kwargs(tools=_MINIMAX_TOOLS, provider="minimax")
        tools = kwargs["tools"]
        assert tools[-1] == _SERVER_ENTRY
        assert len(tools) == 2  # converted function tool + exactly one server entry
        assert "input_schema" not in tools[-1]
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_flag_on_minimax_oauth_provider(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
            kwargs = _kwargs(tools=_MINIMAX_TOOLS, provider="minimax-oauth")
        assert kwargs["tools"][-1] == _SERVER_ENTRY

    def test_flag_on_non_minimax_absent(self):
        for provider in ("anthropic", "openrouter", "kimi", None):
            with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
                kwargs = _kwargs(tools=_MINIMAX_TOOLS, provider=provider)
            assert all(t.get("name") != "web_search" for t in kwargs.get("tools", [])), provider

    def test_flag_on_tool_choice_none_recreates_tools(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
            kwargs = _kwargs(tools=_MINIMAX_TOOLS, tool_choice="none", provider="minimax")
        # tools were popped for "none"; the server entry re-creates the array alone
        assert kwargs["tools"] == [_SERVER_ENTRY]
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_flag_on_forced_tool_choice_becomes_auto(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
            kwargs = _kwargs(tools=_MINIMAX_TOOLS, tool_choice="read_file", provider="minimax")
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["tools"][-1] == _SERVER_ENTRY
        # the forced client tool is still declared
        assert any(t.get("name") == "read_file" for t in kwargs["tools"])

    def test_flag_on_no_tools_no_tool_choice_untouched(self):
        """No tools array being sent and no 'none' to pop => nothing appended."""
        with patch("hermes_cli.config.load_config_readonly", return_value=_config_on()):
            kwargs = _kwargs(tools=None, provider="minimax")
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# normalize_response handling of server_tool_use blocks
# ---------------------------------------------------------------------------

def _server_block(**overrides):
    base = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "weather in Amman"},
        "status": "completed",
        "sources": [
            {"url": "https://example.com/forecast", "title": "Forecast"},
            {"url": "https://example.org/wiki", "title": "Wiki"},
        ],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestNormalizeResponseServerToolUse:
    def test_consumes_block_keeps_text_no_tool_call(self, caplog):
        from agent.transports.anthropic import AnthropicTransport

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="It is sunny today."),
                _server_block(),
            ],
            stop_reason="end_turn",
            stop_details=None,
        )
        with caplog.at_level(logging.DEBUG, logger="agent.transports.anthropic"):
            normalized = AnthropicTransport().normalize_response(response)
        assert normalized.content == "It is sunny today."
        assert normalized.tool_calls is None
        assert normalized.finish_reason == "stop"
        sources = (normalized.provider_data or {}).get("web_search_sources")
        assert sources == [
            {"url": "https://example.com/forecast", "title": "Forecast"},
            {"url": "https://example.org/wiki", "title": "Wiki"},
        ]
        assert any("weather in Amman" in rec.message for rec in caplog.records)
        # replay carrier must not persist the server block (unknown type is dropped)
        assert "anthropic_content_blocks" not in (normalized.provider_data or {})

    def test_inner_web_search_call_shape(self):
        """Also accept the inner-type variant (server_tool_type=web_search_call)."""
        from agent.transports.anthropic import AnthropicTransport

        block = SimpleNamespace(type="server_tool_use", server_tool_type="web_search_call",
                                id="srvtoolu_2", input={"query": "q"}, status="ok",
                                results=[{"url": "https://x.example", "title": "X", "snippet": "s"}])
        response = SimpleNamespace(content=[block], stop_reason="end_turn", stop_details=None)
        normalized = AnthropicTransport().normalize_response(response)
        assert (normalized.provider_data or {}).get("web_search_sources") == [
            {"url": "https://x.example", "title": "X"}
        ]

    def test_preexisting_provider_data_merged_not_replaced(self):
        from agent.transports.anthropic import AnthropicTransport

        response = SimpleNamespace(
            content=[_server_block()],
            stop_reason="end_turn",
            stop_details=None,
            provider_data={"web_search_sources": [{"url": "https://stream.example", "title": "S"}]},
        )
        normalized = AnthropicTransport().normalize_response(response)
        urls = {s["url"] for s in normalized.provider_data["web_search_sources"]}
        assert urls == {"https://stream.example", "https://example.com/forecast", "https://example.org/wiki"}


# ---------------------------------------------------------------------------
# Streaming _call_anthropic
# ---------------------------------------------------------------------------

def _streaming_call_handler(monkeypatch, events, raw_stream, emitted):
    """A _StreamingCall with just enough surface for _call_anthropic, fed by a fake relay stream."""
    from agent import chat_completion_helpers as h
    from agent import relay_llm

    class _FakeRelayStream:
        output_modified = False

        def __init__(self, evts, on_created, on_chunk):
            self._evts, self._on_created, self._on_chunk = evts, on_created, on_chunk

        def __iter__(self):
            self._on_created(raw_stream)
            for ev in self._evts:
                self._on_chunk(ev)
                yield ev

    def _fake_stream(api_kwargs, opener, **kwargs):
        return _FakeRelayStream(events, kwargs["on_stream_created"], kwargs["on_chunk"])

    monkeypatch.setattr(relay_llm, "stream", _fake_stream)

    handler = h._StreamingCall.__new__(h._StreamingCall)
    handler.agent = MagicMock()
    handler.agent._interrupt_requested = False
    handler.agent.log_prefix = ""
    handler.last_chunk_time = {"t": 0.0}
    handler.result = {"partial_tool_names": []}
    handler.managed_stream_holder = {}
    handler._writer_token = None
    handler._attempt_stream_response = None
    handler._attempt_request_client = None
    handler.api_kwargs = {"model": "MiniMax-M2.7", "messages": [], "max_tokens": 10}
    handler._new_diag = dict
    handler._count_chunk = lambda diag, chunk: None
    handler._mark_finish_seen = lambda diag, finish_reason: None
    handler._emit_text = emitted.append
    handler._emit_reasoning = lambda text: None
    handler._emit_tool_started = lambda name: None
    handler._writer_still_current = lambda label: True
    handler._reabort_if_cancelled = lambda response: None
    return handler


class TestStreamingServerToolUse:
    def test_stream_keeps_text_and_collects_sources(self, monkeypatch):
        events = [
            SimpleNamespace(type="message_start", message={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "MiniMax-M2.7",
                "usage": {"input_tokens": 10, "output_tokens": 5}}),
            SimpleNamespace(type="content_block_start", index=0, content_block=_server_block()),
            SimpleNamespace(type="content_block_start", index=1,
                            content_block=SimpleNamespace(type="text", text="")),
            SimpleNamespace(type="content_block_delta", index=1,
                            delta=SimpleNamespace(type="text_delta", text="It is sunny.")),
            SimpleNamespace(type="message_delta", index=2,
                            delta=SimpleNamespace(stop_reason="end_turn", stop_sequence=None)),
            SimpleNamespace(type="message_stop"),
        ]
        final_message = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="It is sunny."),
                _server_block(sources=[{"url": "https://snapshot.example", "title": "Snap"}]),
            ],
            stop_reason="end_turn",
            stop_details=None,
        )
        raw_stream = SimpleNamespace(
            response=None, current_message_snapshot=final_message,
            get_final_message=lambda: final_message,
        )
        emitted: list[str] = []
        handler = _streaming_call_handler(monkeypatch, events, raw_stream, emitted)

        message = handler._call_anthropic(request_client=MagicMock())

        assert message is final_message  # unmodified stream returns the SDK final message
        assert emitted == ["It is sunny."]  # text intact; no tool_use gating kicked in
        sources = (getattr(message, "provider_data", None) or {}).get("web_search_sources") or []
        urls = {s["url"] for s in sources}
        # stream-seen source + final-message snapshot source, deduped
        assert urls == {"https://example.com/forecast", "https://example.org/wiki",
                        "https://snapshot.example"}
        assert handler.result["partial_tool_names"] == []  # server tool never marked as client tool


# ---------------------------------------------------------------------------
# Interleaved replay round-trip
# ---------------------------------------------------------------------------

def _replay_agent(tmp_path):
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api
    from agent.context_compressor import ContextCompressor

    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="MiniMax-M2.7")
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="minimax", model="MiniMax-M2.7",
        base_url="https://api.minimaxi.com/anthropic",
        session_id="s1", _session_db=db, _persist_disabled=False,
        _current_turn_timestamp=1.0, ephemeral_system_prompt="", prefill_messages=[],
        tools=[], _use_prompt_caching=False, _usage_anchor=None, verbose_logging=False,
        reasoning_callback=None, stream_delta_callback=None, _stream_callback=None, log_prefix="",
    )
    agent._needs_thinking_reasoning_pad = lambda: False
    from agent.codex_responses_adapter import _derive_responses_function_call_id
    agent._derive_responses_function_call_id = _derive_responses_function_call_id
    agent._copy_reasoning_content_for_api = (
        lambda source, target: copy_reasoning_content_for_api(agent, source, target)
    )
    agent._should_sanitize_tool_calls = lambda: False
    agent._split_responses_tool_id = lambda raw_id: (raw_id, None)
    agent._sanitize_api_messages = lambda value: value
    agent._emit_warning = agent._vprint = lambda *args, **kwargs: None
    agent._drop_thinking_only_and_merge_users = lambda value, **kwargs: value
    agent._extract_reasoning = lambda message: getattr(message, "reasoning", None)
    agent._strip_think_blocks = lambda text: text
    agent.context_compressor = ContextCompressor(
        "MiniMax-M2.7", provider="minimax", base_url="https://api.minimaxi.com/anthropic",
        api_mode=agent.api_mode, quiet_mode=True, config_context_length=200_000,
    )
    agent.context_compressor.bind_session_state(db, "s1")
    return agent


class TestInterleavedReplayRoundTrip:
    def test_server_tool_use_does_not_misalign_replay(self, tmp_path, caplog):
        """Interleaved turn (signed thinking + server_tool_use + text + client tool_use)
        round-trips: the server block is dropped by the replay sanitizer, the client tool
        input is re-sourced from redacted_input_by_id, and block order is preserved."""
        from agent.anthropic_message_convert import assistant_replay_carrier
        from agent.chat_completion_helpers import build_assistant_message
        from agent.transports.anthropic import AnthropicTransport

        signed = SimpleNamespace(type="thinking", thinking="planning...", signature="sig_1")
        server = _server_block()
        text = SimpleNamespace(type="text", text="Based on the search, calling the tool.")
        tool_use = SimpleNamespace(type="tool_use", id="toolu_1", name="read_file",
                                   input={"path": "/etc/hostname", "leaked": "raw-api-secret"})
        response = SimpleNamespace(content=[signed, server, text, tool_use],
                                   stop_reason="tool_use", stop_details=None)
        with caplog.at_level(logging.DEBUG, logger="agent.transports.anthropic"):
            normalized = AnthropicTransport().normalize_response(response)

        # producer surfaces: exactly one client ToolCall, sources harvested, ordered blocks kept
        assert [tc.name for tc in (normalized.tool_calls or [])] == ["read_file"]
        assert (normalized.provider_data or {})["web_search_sources"][0]["url"] == \
            "https://example.com/forecast"
        ordered = (normalized.provider_data or {}).get("anthropic_content_blocks") or []
        # _REPLAY_SANITIZERS drops the unknown server_tool_use type from the persisted
        # ordered blocks — desired: it is provider output, never valid request input.
        assert [b["type"] for b in ordered] == ["thinking", "text", "tool_use"]

        agent = _replay_agent(tmp_path)
        row = build_assistant_message(agent, normalized, normalized.finish_reason)
        # storage re-sources tool input from the (redacted) tool_calls, not the raw API block
        assert row["tool_calls"][0]["function"]["arguments"] == '{"path": "/etc/hostname", "leaked": "raw-api-secret"}'

        # consumer: the replay carrier drops the server block and keeps everything else aligned
        used_ordered, replayed = assistant_replay_carrier(row)
        assert used_ordered is True
        assert [b["type"] for b in replayed] == ["thinking", "text", "tool_use"]
        assert replayed[0] == {"type": "thinking", "thinking": "planning...", "signature": "sig_1"}
        assert replayed[1] == {"type": "text", "text": "Based on the search, calling the tool."}
        # redacted_input_by_id alignment: input re-sourced by tool id, server id never consulted
        assert replayed[2]["id"] == "toolu_1"
        assert replayed[2]["input"] == {"path": "/etc/hostname", "leaked": "raw-api-secret"}
