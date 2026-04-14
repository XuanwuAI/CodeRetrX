import asyncio

from scripts import bug_hunter_demo as demo


class _FakeQuery:
    def __init__(self, messages):
        self._messages = messages

    async def receive_messages(self):
        for message in self._messages:
            yield message


class _FakeClient:
    def __init__(self, messages):
        self._query = _FakeQuery(messages)


async def _collect_messages(client):
    return [message async for message in demo.receive_response_tolerant(client)]


def test_format_rate_limit_event_includes_key_fields():
    formatted = demo._format_rate_limit_event(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "rejected",
                "utilization": 0.85,
                "rate_limit_type": "burst",
                "resetsAt": "2026-04-14T12:00:00Z",
                "overage_status": "cooldown",
            },
        }
    )

    assert "status=rejected" in formatted
    assert "utilization=85%" in formatted
    assert "type=burst" in formatted
    assert "resets_at=2026-04-14T12:00:00Z" in formatted
    assert "overage_status=cooldown" in formatted


def test_print_message_hides_rate_limit_event_by_default(capsys):
    demo.print_message(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected"},
        }
    )

    stdout = capsys.readouterr().out
    assert stdout == ""


def test_receive_response_tolerant_allows_rate_limit_event_and_stops_at_result():
    client = _FakeClient(
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Scanning"}]},
            },
            {
                "type": "rate_limit_event",
                "rate_limit_info": {"status": "rejected"},
            },
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 10,
                "duration_api_ms": 5,
                "is_error": False,
                "num_turns": 1,
                "session_id": "session_123",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "late"}]},
            },
        ]
    )

    messages = asyncio.run(_collect_messages(client))

    assert [message["type"] for message in messages] == [
        "assistant",
        "rate_limit_event",
        "result",
    ]
