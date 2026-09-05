import json

import pytest

from llm_cost.cli import main


def test_estimate_prints_table(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "12000", "--output", "800"])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "cost per call" in out


def test_estimate_json_matches_breakdown_shape(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "0", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["model"] == "claude-opus-5"
    assert payload["input_tokens"] == 1000
    assert "total_cost" in payload


def test_estimate_json_flag_works_before_subcommand(capsys):
    code = main(["--json", "estimate", "--model", "claude-opus-5", "--input", "1000"])
    out = capsys.readouterr().out
    assert code == 0
    json.loads(out)


def test_estimate_unknown_model_exits_three(capsys):
    code = main(["estimate", "--model", "not-a-real-model", "--input", "10"])
    captured = capsys.readouterr()
    assert code == 3
    assert "llm-cost:" in captured.err


def test_estimate_negative_tokens_exits_two(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "-5"])
    captured = capsys.readouterr()
    assert code == 2
    assert "llm-cost:" in captured.err


def test_models_lists_every_builtin_model(capsys):
    code = main(["models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "provider" in out


def test_models_json(capsys):
    code = main(["models", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "claude-opus-5" in payload
    assert payload["claude-opus-5"]["provider"] == "anthropic"


def test_compare_ranks_cheapest_first(capsys):
    code = main(["compare", "--input", "10000", "--output", "1000", "--provider", "anthropic"])
    out = capsys.readouterr().out
    assert code == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert "claude-haiku-4-5" in lines[2]


def test_compare_json(capsys):
    code = main(["compare", "--input", "1000", "--output", "100", "--json"])
    out = capsys.readouterr().out
    rows = json.loads(out)
    assert rows
    assert rows[0]["ratio"] == pytest.approx(1.0)


def test_compare_unknown_model_in_shortlist_exits_three(capsys):
    code = main(["compare", "--input", "1000", "--output", "100", "--models", "not-a-real-model"])
    captured = capsys.readouterr()
    assert code == 3
    assert "llm-cost:" in captured.err


def test_report_reads_usage_log(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        '{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n'
        '{"model":"internal-router-v3","usage":{"input_tokens":1,"output_tokens":1}}\n',
        encoding="utf-8",
    )
    code = main(["report", str(log)])
    captured = capsys.readouterr()
    assert code == 0
    assert "claude-opus-5" in captured.out
    assert "TOTAL" in captured.out
    assert "malformed record" not in captured.err


def test_report_json(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        '{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n',
        encoding="utf-8",
    )
    code = main(["report", str(log), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["group_by"] == "model"
    assert payload["total_calls"] == 1


def test_report_prints_malformed_lines_to_stderr(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text("not json at all\n", encoding="utf-8")
    code = main(["report", str(log)])
    captured = capsys.readouterr()
    assert code == 0
    assert "malformed record" in captured.err


def test_report_strict_exits_two_on_malformed_line(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text("not json at all\n", encoding="utf-8")
    code = main(["report", str(log), "--strict"])
    captured = capsys.readouterr()
    assert code == 2
    assert "llm-cost:" in captured.err


def test_report_missing_file_exits_two(capsys):
    code = main(["report", "/no/such/file.jsonl"])
    captured = capsys.readouterr()
    assert code == 2
    assert "could not read" in captured.err


def test_pricing_override_accepted_before_and_after_subcommand(tmp_path, capsys):
    pricing_file = tmp_path / "prices.json"
    pricing_file.write_text(
        json.dumps({"as_of": "2026-07-01", "models": {"claude-opus-5": {"input": 1.0, "output": 2.0}}}),
        encoding="utf-8",
    )

    main(["estimate", "--model", "claude-opus-5", "--input", "1000000", "--pricing", str(pricing_file)])
    after = json.loads(
        _capture_json(capsys, ["estimate", "--model", "claude-opus-5", "--input", "1000000", "--pricing", str(pricing_file), "--json"])
    )
    before = json.loads(
        _capture_json(capsys, ["--pricing", str(pricing_file), "estimate", "--model", "claude-opus-5", "--input", "1000000", "--json"])
    )
    assert after["input_cost"] == pytest.approx(1.0)
    assert before["input_cost"] == pytest.approx(1.0)


def _capture_json(capsys, argv):
    main(argv)
    return capsys.readouterr().out


def test_pricing_file_not_found_exits_two(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "10", "--pricing", "/no/such/prices.json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "llm-cost:" in captured.err
