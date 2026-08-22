"""Command-line entry point: estimate, report, compare, models.

``--pricing`` and ``--json`` are accepted both before and after the
subcommand name (``llm-cost --json estimate ...`` and
``llm-cost estimate --json ...`` both work) by declaring them on the top
parser with real defaults and again on each subparser with
``default=argparse.SUPPRESS``, so a subparser only overrides the namespace
when the flag is actually given at that position.
"""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

__all__ = ["main"]


class CliError(Exception):
    """Carries the process exit code alongside the message for stderr."""

    def __init__(self, message, code):
        super(CliError, self).__init__(message)
        self.code = code


def _call_phrase(calls):
    return "%d %s" % (calls, "call" if calls == 1 else "calls")


def _load_pricing_table(args):
    if args.pricing:
        try:
            return load_pricing(args.pricing)
        except ValueError as error:
            raise CliError(str(error), 2)
    return default_pricing()


def _cmd_estimate(args):
    table = _load_pricing_table(args)
    try:
        price = table.resolve(args.model)
    except UnknownModelError as error:
        raise CliError(str(error), 3)
    try:
        breakdown = estimate_cost(
            price,
            input_tokens=args.input,
            output_tokens=args.output,
            cached_input_tokens=args.cached,
            cache_write_tokens=args.cache_write,
            calls=args.calls,
        )
    except ValueError as error:
        raise CliError(str(error), 2)

    if args.json:
        print(json.dumps(breakdown.to_dict()))
        return

    print(
        "%s  (%s, prices as of %s)"
        % (price.model, _call_phrase(breakdown.calls), table.as_of)
    )
    print()
    headers = ["item", "tokens", "$/1M", "cost"]
    rows = [
        ["input", format_int(breakdown.input_tokens), "%.2f" % price.input, format_money(breakdown.input_cost)],
        ["cached input", format_int(breakdown.cached_input_tokens), "%.2f" % price.cached_input, format_money(breakdown.cached_input_cost)],
        ["cache write", format_int(breakdown.cache_write_tokens), "%.2f" % price.cache_write, format_money(breakdown.cache_write_cost)],
        ["output", format_int(breakdown.output_tokens), "%.2f" % price.output, format_money(breakdown.output_cost)],
        ["total", format_int(breakdown.total_tokens), "", format_money(breakdown.total_cost)],
    ]
    print(render_table(headers, rows))
    print()
    print("cost per call: %s" % format_money(breakdown.cost_per_call))


def _cmd_report(args):
    try:
        with open(args.path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise CliError("could not read %s: %s" % (args.path, error), 2)

    table = _load_pricing_table(args)
    try:
        records, problems = load_usage(text, strict=args.strict)
    except ValueError as error:
        raise CliError(str(error), 2)

    report = build_report(records, table, group_by=args.group_by)

    if args.json:
        payload = {
            "group_by": report.group_by,
            "groups": [group.to_dict() for group in report.groups],
            "total_calls": report.total_calls,
            "total_cost": round(report.total_cost, 6),
            "unknown_models": sorted(report.unknown_models),
            "skipped": report.skipped,
            "problems": problems,
        }
        print(json.dumps(payload))
        return

    print(report.render())
    for problem in problems:
        print("malformed record, %s" % problem, file=sys.stderr)


def _cmd_compare(args):
    table = _load_pricing_table(args)
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]
    try:
        rows = compare_models(
            table,
            input_tokens=args.input,
            output_tokens=args.output,
            cached_input_tokens=args.cached,
            cache_write_tokens=args.cache_write,
            calls=args.calls,
            models=models,
            provider=args.provider,
        )
    except UnknownModelError as error:
        raise CliError(str(error), 3)
    except ValueError as error:
        raise CliError(str(error), 2)

    if args.json:
        print(json.dumps(rows))
        return

    print(
        "%s in + %s out, %s, prices as of %s"
        % (
            format_int(args.input),
            format_int(args.output),
            _call_phrase(args.calls),
            table.as_of,
        )
    )
    print()
    headers = ["model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest"]
    body = [
        [
            row["model"],
            row["provider"],
            "%.2f" % row["input_price"],
            "%.2f" % row["output_price"],
            format_money(row["cost"]),
            "%.1fx" % row["ratio"],
        ]
        for row in rows
    ]
    print(render_table(headers, body))


def _cmd_models(args):
    table = _load_pricing_table(args)

    if args.json:
        payload = dict((name, table.prices[name].to_dict()) for name in table.models())
        print(json.dumps(payload))
        return

    print(
        "%d models, USD per 1M tokens, as of %s (source: %s)"
        % (len(table), table.as_of, table.source)
    )
    print()
    headers = ["model", "provider", "input", "output", "cached_input", "cache_write"]
    body = []
    for name in table.models():
        price = table.prices[name]
        body.append(
            [
                price.model,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                "%.2f" % price.cached_input,
                "%.2f" % price.cache_write,
            ]
        )
    print(render_table(headers, body))


def _add_global_options(parser, suppress_defaults):
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--pricing", metavar="FILE", default=default, help="JSON file overriding the built-in price table")
    json_default = argparse.SUPPRESS if suppress_defaults else False
    parser.add_argument("--json", action="store_true", default=json_default, help="machine-readable output")


def _build_parser():
    parser = argparse.ArgumentParser(prog="llm-cost", description=__doc__.split("\n")[0])
    _add_global_options(parser, suppress_defaults=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="cost of one call or a batch of identical calls")
    _add_global_options(estimate, suppress_defaults=True)
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--input", type=int, default=0, metavar="TOKENS")
    estimate.add_argument("--output", type=int, default=0, metavar="TOKENS")
    estimate.add_argument("--cached", type=int, default=0, metavar="TOKENS", help="cache-read tokens")
    estimate.add_argument("--cache-write", type=int, default=0, metavar="TOKENS")
    estimate.add_argument("--calls", type=int, default=1)
    estimate.set_defaults(func=_cmd_estimate)

    report = subparsers.add_parser("report", help="cost of a JSONL usage log, grouped and totalled")
    _add_global_options(report, suppress_defaults=True)
    report.add_argument("path", help="JSONL usage log, one API call per line")
    report.add_argument("--group-by", default="model", metavar="FIELD")
    report.add_argument("--strict", action="store_true", help="fail on the first malformed line instead of skipping it")
    report.set_defaults(func=_cmd_report)

    compare = subparsers.add_parser("compare", help="rank models cheapest first for a given call shape")
    _add_global_options(compare, suppress_defaults=True)
    compare.add_argument("--input", type=int, default=0, metavar="TOKENS")
    compare.add_argument("--output", type=int, default=0, metavar="TOKENS")
    compare.add_argument("--cached", type=int, default=0, metavar="TOKENS", help="cache-read tokens")
    compare.add_argument("--cache-write", type=int, default=0, metavar="TOKENS")
    compare.add_argument("--calls", type=int, default=1)
    compare.add_argument("--models", metavar="A,B,C", help="comma-separated shortlist, default is every priced model")
    compare.add_argument("--provider", help="restrict to one provider, e.g. anthropic")
    compare.set_defaults(func=_cmd_compare)

    models = subparsers.add_parser("models", help="list every model in the active price table")
    _add_global_options(models, suppress_defaults=True)
    models.set_defaults(func=_cmd_models)

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except CliError as error:
        print("llm-cost: %s" % error, file=sys.stderr)
        return error.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
