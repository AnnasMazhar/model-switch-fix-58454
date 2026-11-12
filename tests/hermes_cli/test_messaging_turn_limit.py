"""Chat turns can carry a tighter iteration cap than cron.

Each tool iteration replays the transcript. DeepSeek bills that replay as
cache-read tokens, so a 50-step chat turn is the measured cost, not the model.
"""

import pytest

from hermes_cli.config import TURN_LIMIT_UNLIMITED, apply_messaging_turn_cap


def test_missing_key_keeps_the_cron_budget():
    assert apply_messaging_turn_cap({"max_turns": 50}, 50) == 50


def test_explicit_cap_replaces_the_base():
    assert apply_messaging_turn_cap({"messaging_max_turns": 8, "max_turns": 50}, 50) == 8


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 50),
        (True, 50),
        ("unlimited", TURN_LIMIT_UNLIMITED),
        ("none", TURN_LIMIT_UNLIMITED),
        ("0", TURN_LIMIT_UNLIMITED),
        ("8", 8),
        (3, 3),
        ("nope", 50),
        ([], 50),
    ],
)
def test_messaging_cap_mutations(raw, expected):
    # Mutate the same dict the gateway would have loaded. The base stays the
    # cron budget whenever the chat cap is absent or unparseable.
    cfg = {"max_turns": 50}
    cfg["messaging_max_turns"] = raw
    assert apply_messaging_turn_cap(cfg, 50) == expected


def test_non_dict_config_is_ignored():
    assert apply_messaging_turn_cap(None, 12) == 12
    assert apply_messaging_turn_cap(["messaging_max_turns"], 12) == 12


def test_gateway_chat_budget_uses_the_messaging_cap(monkeypatch):
    from gateway.run import _bridge_max_turns_to_env, _current_max_iterations

    monkeypatch.setattr("gateway.run._reload_runtime_env_preserving_config_authority", lambda: None)
    monkeypatch.setattr("gateway.run.get_hermes_home_override", lambda: None)
    monkeypatch.delenv("HERMES_MESSAGING_MAX_ITERATIONS", raising=False)
    _bridge_max_turns_to_env({"max_turns": 50, "messaging_max_turns": 8})
    assert _current_max_iterations() == 8
