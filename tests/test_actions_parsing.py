import json

from context_agent.agent.actions import ActionParseError, AgentAction, parse_action


def test_parse_valid_tool_action():
    raw = json.dumps({"thought": "search first", "tool": "search_corpus", "arguments": {"query": "x"}})
    action = parse_action(raw)
    assert isinstance(action, AgentAction)
    assert action.tool.value == "search_corpus"
    assert action.arguments == {"query": "x"}
    assert not action.finish


def test_parse_finish_action():
    raw = json.dumps({"thought": "done", "finish": True})
    action = parse_action(raw)
    assert isinstance(action, AgentAction)
    assert action.finish is True
    assert action.tool is None


def test_parse_tolerates_surrounding_text():
    raw = 'Sure, here is my action:\n{"thought": "ok", "finish": true}\nHope that helps.'
    action = parse_action(raw)
    assert isinstance(action, AgentAction)
    assert action.finish is True


def test_parse_invalid_json_returns_error():
    action = parse_action("not json at all {broken")
    assert isinstance(action, ActionParseError)


def test_parse_no_json_object_returns_error():
    action = parse_action("I don't know what to do")
    assert isinstance(action, ActionParseError)


def test_parse_unknown_tool_returns_error():
    raw = json.dumps({"thought": "x", "tool": "delete_everything", "arguments": {}})
    action = parse_action(raw)
    assert isinstance(action, ActionParseError)
    assert "unknown tool" in action.error


def test_parse_missing_tool_and_not_finished_returns_error():
    raw = json.dumps({"thought": "x"})
    action = parse_action(raw)
    assert isinstance(action, ActionParseError)


def test_parse_non_object_json_returns_error():
    action = parse_action("[1, 2, 3]")
    assert isinstance(action, ActionParseError)
