"""Plain-text table rendering shared by the report and CLI output.

Column widths are derived from the data itself rather than fixed, since model
names and dollar totals vary wildly in width between a single estimate and a
month of usage logs.
"""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(value):
    """Thousands-grouped integer, e.g. ``12000`` -> ``"12,000"``."""
    return "{:,}".format(value)


def format_money(amount):
    """Dollar amount at four decimal places, e.g. ``0.06`` -> ``"$0.0600"``.

    Four places rather than the usual two because per-call costs routinely
    round to zero at two decimals while still summing to a real total.
    """
    return "$" + format(amount, ",.4f")


def render_table(headers, rows):
    """Render ``headers`` and ``rows`` as an aligned plain-text table.

    The first column is left-justified (it usually holds a name); every other
    column is right-justified (they usually hold numbers). Cells are taken as
    already-formatted strings -- this function only aligns, it does not know
    about money or counts.
    """
    headers = [str(cell) for cell in headers]
    body = [[str(cell) for cell in row] for row in rows]

    widths = []
    for index, header in enumerate(headers):
        column = [header] + [row[index] for row in body if index < len(row)]
        widths.append(max(len(value) for value in column))

    def render_row(cells):
        parts = []
        for index, cell in enumerate(cells):
            width = widths[index]
            parts.append(cell.ljust(width) if index == 0 else cell.rjust(width))
        return "  ".join(parts)

    lines = [render_row(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render_row(row) for row in body)
    return "\n".join(lines)
