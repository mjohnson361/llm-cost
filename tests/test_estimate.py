import pytest

from llm_cost.estimate import CostBreakdown, estimate_cost
from llm_cost.pricing import ModelPrice

PRICE = ModelPrice(
    "m", input=2.0, output=8.0, cached_input=0.2, cache_write=2.5, provider="acme"
)


def test_estimate_cost_prices_each_token_class():
    breakdown = estimate_cost(
        PRICE,
        input_tokens=1000000,
        output_tokens=1000000,
        cached_input_tokens=1000000,
        cache_write_tokens=1000000,
    )
    assert breakdown.input_cost == pytest.approx(2.0)
    assert breakdown.output_cost == pytest.approx(8.0)
    assert breakdown.cached_input_cost == pytest.approx(0.2)
    assert breakdown.cache_write_cost == pytest.approx(2.5)


def test_estimate_cost_defaults_to_zero_tokens_and_one_call():
    breakdown = estimate_cost(PRICE)
    assert breakdown.calls == 1
    assert breakdown.total_tokens == 0
    assert breakdown.total_cost == 0.0


def test_estimate_cost_scales_token_counts_by_calls():
    breakdown = estimate_cost(PRICE, input_tokens=100, output_tokens=50, calls=3)
    assert breakdown.input_tokens == 300
    assert breakdown.output_tokens == 150
    assert breakdown.calls == 3


def test_estimate_cost_scales_cost_by_calls():
    one_call = estimate_cost(PRICE, input_tokens=100, output_tokens=50, calls=1)
    three_calls = estimate_cost(PRICE, input_tokens=100, output_tokens=50, calls=3)
    assert three_calls.total_cost == pytest.approx(one_call.total_cost * 3)


def test_estimate_cost_carries_model_name_from_price():
    breakdown = estimate_cost(PRICE, input_tokens=100)
    assert breakdown.model == "m"


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=-1)


def test_estimate_cost_rejects_negative_calls():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, calls=-1)


def test_estimate_cost_rejects_non_integer_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=1.5)


def test_cost_breakdown_total_tokens_sums_all_classes():
    breakdown = CostBreakdown(
        model="m",
        calls=1,
        input_tokens=10,
        output_tokens=20,
        cached_input_tokens=30,
        cache_write_tokens=40,
        input_cost=0.0,
        output_cost=0.0,
        cached_input_cost=0.0,
        cache_write_cost=0.0,
    )
    assert breakdown.total_tokens == 100


def test_cost_breakdown_total_cost_sums_all_classes():
    breakdown = CostBreakdown(
        model="m",
        calls=1,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_tokens=0,
        input_cost=1.0,
        output_cost=2.0,
        cached_input_cost=3.0,
        cache_write_cost=4.0,
    )
    assert breakdown.total_cost == 10.0


def test_cost_breakdown_cost_per_call_divides_by_calls():
    breakdown = CostBreakdown(
        model="m",
        calls=4,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_tokens=0,
        input_cost=8.0,
        output_cost=0.0,
        cached_input_cost=0.0,
        cache_write_cost=0.0,
    )
    assert breakdown.cost_per_call == 2.0


def test_cost_breakdown_cost_per_call_is_zero_when_no_calls():
    breakdown = CostBreakdown(
        model="m",
        calls=0,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_tokens=0,
        input_cost=0.0,
        output_cost=0.0,
        cached_input_cost=0.0,
        cache_write_cost=0.0,
    )
    assert breakdown.cost_per_call == 0.0


def test_cost_breakdown_to_dict_rounds_costs_to_six_places():
    breakdown = estimate_cost(PRICE, input_tokens=1, output_tokens=1)
    as_dict = breakdown.to_dict()
    assert as_dict["model"] == "m"
    assert as_dict["input_tokens"] == 1
    assert as_dict["total_cost"] == round(breakdown.total_cost, 6)


def test_cost_breakdown_repr_includes_model_and_total():
    breakdown = estimate_cost(PRICE, input_tokens=1000000, output_tokens=0)
    text = repr(breakdown)
    assert "m" in text
    assert "2.000000" in text
