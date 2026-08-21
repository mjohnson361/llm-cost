import pytest

from llm_cost.pricing import default_pricing
from llm_cost.report import build_report, compare_models
from llm_cost.usage import load_usage

USAGE = "\n".join(
    [
        '{"model":"claude-opus-5","team":"agents","usage":{"input_tokens":18400,"output_tokens":2100,"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}',
        '{"model":"claude-opus-5","team":"agents","usage":{"input_tokens":9200,"output_tokens":1400,"cache_read_input_tokens":52000}}',
        '{"model":"claude-haiku-4-5","team":"search","usage":{"input_tokens":140000,"output_tokens":9800}}',
        '{"model":"gemini-2.5-flash","team":"ops","usage":{"input_tokens":88000,"output_tokens":3100}}',
        '{"model":"gpt-4o-mini","team":"search","usage":{"prompt_tokens":31000,"completion_tokens":420,"prompt_tokens_details":{"cached_tokens":24000}}}',
        '{"model":"internal-router-v3","team":"agents","usage":{"input_tokens":500,"output_tokens":50}}',
    ]
)


def _records():
    records, problems = load_usage(USAGE)
    assert problems == []
    return records


def test_build_report_groups_by_model_and_sums_tokens():
    report = build_report(_records(), default_pricing(), group_by="model")
    by_model = dict((group.key, group) for group in report.groups)

    assert by_model["claude-opus-5"].calls == 2
    assert by_model["claude-opus-5"].input_tokens == 27600
    assert by_model["claude-opus-5"].cached_input_tokens == 104000
    assert by_model["claude-opus-5"].output_tokens == 3500


def test_build_report_orders_groups_most_expensive_first():
    report = build_report(_records(), default_pricing(), group_by="model")
    costs = [group.cost for group in report.groups]
    assert costs == sorted(costs, reverse=True)
    assert report.groups[0].key == "claude-opus-5"


def test_build_report_sets_aside_records_with_no_price():
    report = build_report(_records(), default_pricing(), group_by="model")
    assert report.skipped == 1
    assert report.unknown_models == {"internal-router-v3"}
    assert "internal-router-v3" not in [group.key for group in report.groups]


def test_build_report_groups_by_arbitrary_field():
    report = build_report(_records(), default_pricing(), group_by="team")
    keys = [group.key for group in report.groups]
    assert set(keys) == {"agents", "search", "ops"}


def test_build_report_totals_match_group_sums():
    report = build_report(_records(), default_pricing(), group_by="model")
    assert report.total_calls == sum(group.calls for group in report.groups)
    assert report.total_cost == pytest.approx(sum(group.cost for group in report.groups))


def test_report_render_includes_total_row_and_skip_notice():
    report = build_report(_records(), default_pricing(), group_by="model")
    rendered = report.render()
    lines = rendered.split("\n")
    assert lines[0].startswith("model ")
    assert any(line.startswith("TOTAL") for line in lines)
    assert rendered.endswith("skipped 1 record(s) with no price: internal-router-v3")


def test_report_render_omits_skip_notice_when_nothing_skipped():
    records, _ = load_usage(
        '{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":1}}'
    )
    report = build_report(records, default_pricing())
    assert "skipped" not in report.render()


def test_compare_models_ranks_cheapest_first():
    rows = compare_models(
        default_pricing(), input_tokens=10000, output_tokens=1000, provider="anthropic"
    )
    assert [row["model"] for row in rows][0] == "claude-haiku-4-5"
    costs = [row["cost"] for row in rows]
    assert costs == sorted(costs)


def test_compare_models_ratio_is_relative_to_cheapest():
    rows = compare_models(
        default_pricing(), input_tokens=10000, output_tokens=1000, provider="anthropic"
    )
    assert rows[0]["ratio"] == pytest.approx(1.0)
    assert rows[-1]["ratio"] == pytest.approx(rows[-1]["cost"] / rows[0]["cost"])


def test_compare_models_filters_by_provider():
    rows = compare_models(
        default_pricing(), input_tokens=1000, output_tokens=100, provider="google"
    )
    assert rows
    assert all(row["provider"] == "google" for row in rows)


def test_compare_models_narrows_to_shortlist():
    rows = compare_models(
        default_pricing(),
        input_tokens=1000,
        output_tokens=100,
        models=["claude-opus-5", "gpt-4o-mini"],
    )
    assert set(row["model"] for row in rows) == {"claude-opus-5", "gpt-4o-mini"}


def test_compare_models_raises_for_unknown_model_in_shortlist():
    with pytest.raises(Exception):
        compare_models(
            default_pricing(),
            input_tokens=1000,
            output_tokens=100,
            models=["not-a-real-model"],
        )
