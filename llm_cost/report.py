"""Aggregate usage records into per-group cost summaries.

Grouping happens on whatever field the usage log carries (model, team, date,
...); pricing happens per record, since two records in the same group can
still name different models. A record naming a model with no price is never
folded into a group as zero -- it is set aside and reported by name so an
unpriced model shows up as a gap, not a silent undercount.
"""

from .estimate import estimate_cost
from .pricing import UnknownModelError
from .table import format_int, format_money, render_table
from .usage import aggregate

__all__ = ["GroupSummary", "Report", "build_report", "compare_models"]


class GroupSummary(object):
    """Cost of every record in one group (one model, one team, one day, ...)."""

    __slots__ = (
        "key",
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "cost",
    )

    def __init__(
        self,
        key,
        calls,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        cost,
    ):
        self.key = key
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.cost = cost

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.cost / self.calls

    def to_dict(self):
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost": round(self.cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }

    def _row(self):
        return [
            self.key,
            format_int(self.calls),
            format_int(self.input_tokens),
            format_int(self.cached_input_tokens),
            format_int(self.output_tokens),
            format_money(self.cost),
            format_money(self.cost_per_call),
        ]


class Report(object):
    """Usage grouped by ``group_by``, ranked most expensive group first."""

    __slots__ = ("group_by", "groups", "unknown_models", "skipped")

    def __init__(self, group_by, groups, unknown_models, skipped):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.skipped = skipped

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_input_tokens(self):
        return sum(group.input_tokens for group in self.groups)

    @property
    def total_output_tokens(self):
        return sum(group.output_tokens for group in self.groups)

    @property
    def total_cached_input_tokens(self):
        return sum(group.cached_input_tokens for group in self.groups)

    @property
    def total_cache_write_tokens(self):
        return sum(group.cache_write_tokens for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.cost for group in self.groups)

    def render(self):
        """The plain-text table shown in the README, TOTAL row included."""
        headers = [self.group_by, "calls", "input", "cached", "output", "cost", "$/call"]
        rows = [group._row() for group in self.groups]
        rows.append(
            [
                "TOTAL",
                format_int(self.total_calls),
                format_int(self.total_input_tokens),
                format_int(self.total_cached_input_tokens),
                format_int(self.total_output_tokens),
                format_money(self.total_cost),
                "",
            ]
        )
        lines = [render_table(headers, rows)]
        if self.skipped:
            lines.append("")
            lines.append(
                "skipped %d record(s) with no price: %s"
                % (self.skipped, ", ".join(sorted(self.unknown_models)))
            )
        return "\n".join(lines)


def build_report(records, table, group_by="model"):
    """Group ``records`` by ``group_by`` and cost each group against ``table``.

    A record naming a model with no entry in ``table`` is excluded from every
    group; it is counted in ``report.skipped`` and its model name added to
    ``report.unknown_models`` instead. Groups are ordered most expensive
    first, which is usually the order worth looking at.
    """
    priced = []
    unknown_models = set()
    skipped = 0
    for record in records:
        try:
            table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            skipped += 1
            continue
        priced.append(record)

    groups = []
    for key, group_records in aggregate(priced, group_by=group_by).items():
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cache_write_tokens = 0
        for record in group_records:
            breakdown = estimate_cost(
                table.resolve(record.model),
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cached_input_tokens=record.cached_input_tokens,
                cache_write_tokens=record.cache_write_tokens,
            )
            cost += breakdown.total_cost
            input_tokens += record.input_tokens
            output_tokens += record.output_tokens
            cached_input_tokens += record.cached_input_tokens
            cache_write_tokens += record.cache_write_tokens
        groups.append(
            GroupSummary(
                key=key,
                calls=len(group_records),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_write_tokens=cache_write_tokens,
                cost=cost,
            )
        )
    groups.sort(key=lambda group: group.cost, reverse=True)
    return Report(group_by, groups, unknown_models, skipped)


def compare_models(
    table,
    input_tokens,
    output_tokens,
    cached_input_tokens=0,
    cache_write_tokens=0,
    calls=1,
    models=None,
    provider=None,
):
    """Cost ``input_tokens``/``output_tokens`` against every model in ``table``.

    Returns a list of dicts sorted cheapest first, each holding the model's
    prices, its cost, and ``ratio`` -- the cost as a multiple of the cheapest
    model's. Narrow the field with ``models`` (an iterable of names, each
    resolved the same way ``estimate`` resolves one) or ``provider`` (matched
    exactly against each price's ``provider``).
    """
    names = models if models is not None else table.models()
    rows = []
    for name in names:
        price = table.resolve(name)
        if provider and price.provider != provider:
            continue
        breakdown = estimate_cost(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            calls=calls,
        )
        rows.append(
            {
                "model": price.model,
                "provider": price.provider,
                "input_price": price.input,
                "output_price": price.output,
                "cost": breakdown.total_cost,
            }
        )
    rows.sort(key=lambda row: row["cost"])
    cheapest = rows[0]["cost"] if rows else 0.0
    for row in rows:
        row["ratio"] = (row["cost"] / cheapest) if cheapest else 1.0
    return rows
