"""Parsing for OpenAI- and Anthropic-shaped JSONL usage logs.

The two providers disagree about what "input tokens" means on the wire.
OpenAI's ``prompt_tokens`` includes any cached prefix; the cached portion is
broken out separately under ``prompt_tokens_details.cached_tokens``.
Anthropic's ``input_tokens`` already excludes the cached prefix, which is
reported instead as ``cache_read_input_tokens`` (and cache writes as
``cache_creation_input_tokens``). Everything in this module normalises to the
exclusive convention -- ``input_tokens`` never includes a cached token -- so
the rest of the package only has to handle one shape.
"""

import json

__all__ = ["UsageRecord", "parse_record", "load_usage", "aggregate"]


class UsageRecord(object):
    """One parsed and normalised usage-log line.

    ``fields`` holds every top-level key from the source record (including
    ``model``, plus things like ``date`` or ``team``) so callers can group by
    any of them without this module having to know what they mean.
    """

    __slots__ = (
        "model",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "fields",
    )

    def __init__(
        self,
        model,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        fields,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.fields = fields

    def field(self, name):
        """Raw value of top-level field ``name`` from the source record, or ``None``."""
        return self.fields.get(name)

    def __repr__(self):
        return "UsageRecord(%s, in=%d, out=%d)" % (
            self.model,
            self.input_tokens,
            self.output_tokens,
        )


def _non_negative_int(name, value, model):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s for model %r must be an integer" % (name, model))
    if value < 0:
        raise ValueError("%s for model %r must not be negative" % (name, model))
    return value


def parse_record(obj):
    """Parse one decoded JSON object into a :class:`UsageRecord`.

    Raises :class:`ValueError` if the record has no model, no usage object, or
    a usage shape this module does not recognise.
    """
    if not isinstance(obj, dict):
        raise ValueError("usage record must be a JSON object")
    model = obj.get("model")
    if not model:
        raise ValueError("usage record has no 'model' field")
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage record for %r has no 'usage' object" % model)

    if "prompt_tokens" in usage:
        # OpenAI: prompt_tokens is inclusive of the cached prefix.
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details")
        cached_input_tokens = 0
        if isinstance(details, dict):
            cached_input_tokens = details.get("cached_tokens", 0)
        input_tokens = prompt_tokens - cached_input_tokens
        cache_write_tokens = 0
    elif "input_tokens" in usage:
        # Anthropic: input_tokens already excludes the cached prefix.
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_input_tokens = usage.get("cache_read_input_tokens", 0)
        cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
    else:
        raise ValueError(
            "usage record for %r has neither 'prompt_tokens' nor 'input_tokens'"
            % model
        )

    input_tokens = _non_negative_int("input_tokens", input_tokens, model)
    output_tokens = _non_negative_int("output_tokens", output_tokens, model)
    cached_input_tokens = _non_negative_int(
        "cached_input_tokens", cached_input_tokens, model
    )
    cache_write_tokens = _non_negative_int(
        "cache_write_tokens", cache_write_tokens, model
    )

    fields = dict((key, value) for key, value in obj.items() if key != "usage")

    return UsageRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        fields=fields,
    )


def load_usage(text, strict=False):
    """Parse ``text`` as JSONL usage records.

    Returns ``(records, problems)``. Blank lines are skipped. A line that is
    not valid JSON, or decodes to a record this module cannot parse, is
    recorded in ``problems`` as ``"line N: <reason>"`` and otherwise ignored --
    one bad row should not lose the rest of a log. Pass ``strict=True`` to
    raise on the first such line instead.
    """
    records = []
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            record = parse_record(obj)
        except ValueError as error:
            message = "line %d: %s" % (number, error)
            if strict:
                raise ValueError(message)
            problems.append(message)
            continue
        records.append(record)
    return records, problems


def aggregate(records, group_by="model"):
    """Group ``records`` by top-level field ``group_by``.

    Returns a dict mapping the stringified field value to the list of records
    sharing it, in first-seen order. Records missing the field are grouped
    under ``"(none)"`` rather than dropped.
    """
    groups = {}
    for record in records:
        key = record.field(group_by)
        key = "(none)" if key is None else str(key)
        groups.setdefault(key, []).append(record)
    return groups
