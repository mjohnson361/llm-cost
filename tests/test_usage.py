import pytest

from llm_cost.usage import aggregate, load_usage, parse_record


def test_parse_record_anthropic_shape():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "date": "2026-06-01T09:12:00Z",
            "team": "agents",
            "usage": {
                "input_tokens": 18400,
                "output_tokens": 2100,
                "cache_read_input_tokens": 52000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    assert record.model == "claude-opus-5"
    assert record.input_tokens == 18400
    assert record.output_tokens == 2100
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000
    assert record.field("team") == "agents"
    assert record.field("date") == "2026-06-01T09:12:00Z"


def test_parse_record_anthropic_shape_defaults_missing_cache_fields_to_zero():
    record = parse_record(
        {"model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 10}}
    )
    assert record.cached_input_tokens == 0
    assert record.cache_write_tokens == 0


def test_parse_record_openai_shape_subtracts_cached_from_prompt_tokens():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "team": "search",
            "usage": {
                "prompt_tokens": 31000,
                "completion_tokens": 420,
                "prompt_tokens_details": {"cached_tokens": 24000},
            },
        }
    )
    assert record.input_tokens == 7000
    assert record.output_tokens == 420
    assert record.cached_input_tokens == 24000
    assert record.cache_write_tokens == 0


def test_parse_record_openai_shape_without_cache_details():
    record = parse_record(
        {"model": "gpt-4o", "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}
    )
    assert record.input_tokens == 1000
    assert record.cached_input_tokens == 0


def test_parse_record_keeps_unknown_top_level_fields():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "region": "us-east-1",
        }
    )
    assert record.field("region") == "us-east-1"
    assert record.field("missing") is None


def test_parse_record_requires_model():
    with pytest.raises(ValueError):
        parse_record({"usage": {"input_tokens": 1, "output_tokens": 1}})


def test_parse_record_requires_usage_object():
    with pytest.raises(ValueError):
        parse_record({"model": "claude-opus-5"})


def test_parse_record_rejects_unrecognised_usage_shape():
    with pytest.raises(ValueError):
        parse_record({"model": "claude-opus-5", "usage": {"tokens": 100}})


def test_parse_record_rejects_negative_token_counts():
    with pytest.raises(ValueError):
        parse_record(
            {"model": "claude-opus-5", "usage": {"input_tokens": -1, "output_tokens": 1}}
        )


def test_parse_record_rejects_non_integer_token_counts():
    with pytest.raises(ValueError):
        parse_record(
            {"model": "claude-opus-5", "usage": {"input_tokens": 1.5, "output_tokens": 1}}
        )


def test_load_usage_parses_multiple_lines():
    text = "\n".join(
        [
            '{"model":"claude-opus-5","usage":{"input_tokens":10,"output_tokens":2}}',
            '{"model":"gpt-4o-mini","usage":{"prompt_tokens":5,"completion_tokens":1}}',
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 2
    assert problems == []
    assert [record.model for record in records] == ["claude-opus-5", "gpt-4o-mini"]


def test_load_usage_skips_blank_lines():
    text = '\n\n{"model":"claude-opus-5","usage":{"input_tokens":10,"output_tokens":2}}\n\n'
    records, problems = load_usage(text)
    assert len(records) == 1
    assert problems == []


def test_load_usage_collects_problems_for_bad_lines_by_default():
    text = "\n".join(
        [
            "not json at all",
            '{"model":"claude-opus-5","usage":{"input_tokens":10,"output_tokens":2}}',
            '{"model":"internal-router-v3"}',
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 2
    assert problems[0].startswith("line 1:")
    assert problems[1].startswith("line 3:")


def test_load_usage_strict_raises_on_first_bad_line():
    text = "\n".join(
        [
            '{"model":"claude-opus-5","usage":{"input_tokens":10,"output_tokens":2}}',
            "not json at all",
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        load_usage(text, strict=True)
    assert "line 2" in str(excinfo.value)


def test_aggregate_groups_by_field_in_first_seen_order():
    text = "\n".join(
        [
            '{"model":"claude-opus-5","team":"agents","usage":{"input_tokens":1,"output_tokens":1}}',
            '{"model":"gpt-4o-mini","team":"search","usage":{"prompt_tokens":1,"completion_tokens":1}}',
            '{"model":"claude-haiku-4-5","team":"agents","usage":{"input_tokens":1,"output_tokens":1}}',
        ]
    )
    records, _ = load_usage(text)
    groups = aggregate(records, group_by="team")
    assert list(groups.keys()) == ["agents", "search"]
    assert [record.model for record in groups["agents"]] == [
        "claude-opus-5",
        "claude-haiku-4-5",
    ]
    assert [record.model for record in groups["search"]] == ["gpt-4o-mini"]


def test_aggregate_groups_missing_field_as_none_bucket():
    text = '{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":1}}'
    records, _ = load_usage(text)
    groups = aggregate(records, group_by="team")
    assert list(groups.keys()) == ["(none)"]


def test_aggregate_default_groups_by_model():
    text = "\n".join(
        [
            '{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":1}}',
            '{"model":"claude-opus-5","usage":{"input_tokens":2,"output_tokens":2}}',
        ]
    )
    records, _ = load_usage(text)
    groups = aggregate(records)
    assert list(groups.keys()) == ["claude-opus-5"]
    assert len(groups["claude-opus-5"]) == 2
