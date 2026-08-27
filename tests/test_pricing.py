import os
import tempfile

import pytest

from llm_cost.pricing import (
    BUILTIN_PRICING,
    ModelPrice,
    PricingTable,
    UnknownModelError,
    default_pricing,
    load_pricing,
    parse_pricing,
)


def test_model_price_defaults_cache_fields_to_input_price():
    price = ModelPrice("m", input=2.0, output=8.0)
    assert price.cached_input == 2.0
    assert price.cache_write == 2.0


def test_model_price_accepts_explicit_cache_fields():
    price = ModelPrice("m", input=2.0, output=8.0, cached_input=0.2, cache_write=2.5)
    assert price.cached_input == 0.2
    assert price.cache_write == 2.5


def test_model_price_rejects_negative_price():
    with pytest.raises(ValueError):
        ModelPrice("m", input=-1.0, output=8.0)


def test_model_price_to_dict():
    price = ModelPrice("m", input=2.0, output=8.0, provider="acme")
    assert price.to_dict() == {
        "provider": "acme",
        "input": 2.0,
        "output": 8.0,
        "cached_input": 2.0,
        "cache_write": 2.0,
    }


def test_default_pricing_has_builtin_models():
    table = default_pricing()
    assert len(table) == len(BUILTIN_PRICING)
    assert table.source == "built-in"
    assert "claude-opus-5" in table


def test_pricing_table_resolve_exact_name():
    table = default_pricing()
    price = table.resolve("claude-opus-5")
    assert price.model == "claude-opus-5"


def test_pricing_table_resolve_is_case_insensitive():
    table = default_pricing()
    assert table.resolve("Claude-Opus-5").model == "claude-opus-5"


def test_pricing_table_resolve_strips_provider_prefix():
    table = default_pricing()
    assert table.resolve("anthropic.claude-opus-5").model == "claude-opus-5"
    assert table.resolve("openai/gpt-4o-mini").model == "gpt-4o-mini"


def test_pricing_table_resolve_strips_date_suffix_by_longest_match():
    table = default_pricing()
    assert table.resolve("gpt-4o-mini-2026-01-31").model == "gpt-4o-mini"
    # gpt-4o-mini is a prefix of the name below too, but gpt-4o is not a key,
    # so this should not be confused with a different, shorter model.
    assert table.resolve("gpt-4o-2026-01-31").model == "gpt-4o"


def test_pricing_table_resolve_raises_for_unknown_model():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("not-a-real-model")


def test_pricing_table_resolve_raises_for_empty_model():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("")


def test_pricing_table_contains_uses_resolve():
    table = default_pricing()
    assert "anthropic.claude-opus-5" in table
    assert "not-a-real-model" not in table


def test_pricing_table_models_is_sorted():
    table = default_pricing()
    assert table.models() == sorted(table.models())


def test_parse_pricing_flat_shape_merges_onto_builtin():
    table = parse_pricing({"my-model": {"input": 1, "output": 3}})
    assert table.as_of == "unspecified"
    assert "claude-opus-5" in table
    assert table.resolve("my-model").input == 1.0


def test_parse_pricing_wrapped_shape_reads_as_of():
    table = parse_pricing(
        {"as_of": "2026-07-01", "models": {"my-model": {"input": 1, "output": 3}}}
    )
    assert table.as_of == "2026-07-01"


def test_parse_pricing_replace_drops_builtin_models():
    table = parse_pricing(
        {"replace": True, "models": {"my-model": {"input": 1, "output": 3}}}
    )
    assert "claude-opus-5" not in table
    assert len(table) == 1


def test_parse_pricing_entry_falls_back_to_input_for_cache_fields():
    table = parse_pricing({"my-model": {"input": 1, "output": 3}})
    price = table.resolve("my-model")
    assert price.cached_input == 1.0
    assert price.cache_write == 1.0


def test_parse_pricing_rejects_non_object():
    with pytest.raises(ValueError):
        parse_pricing(["not", "an", "object"])


def test_parse_pricing_rejects_empty_models():
    with pytest.raises(ValueError):
        parse_pricing({"models": {}})


def test_parse_pricing_rejects_entry_missing_required_fields():
    with pytest.raises(ValueError):
        parse_pricing({"my-model": {"input": 1}})


def test_parse_pricing_rejects_non_numeric_price():
    with pytest.raises(ValueError):
        parse_pricing({"my-model": {"input": "cheap", "output": 3}})


def test_parse_pricing_source_defaults_to_override():
    table = parse_pricing({"my-model": {"input": 1, "output": 3}})
    assert table.source == "override"


def test_load_pricing_reads_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"my-model": {"input": 1, "output": 3}}')
        table = load_pricing(path)
        assert table.resolve("my-model").input == 1.0
        assert table.source == path
    finally:
        os.remove(path)


def test_load_pricing_raises_for_missing_file():
    with pytest.raises(ValueError):
        load_pricing("/no/such/pricing/file.json")


def test_load_pricing_raises_for_invalid_json():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not json")
        with pytest.raises(ValueError):
            load_pricing(path)
    finally:
        os.remove(path)
