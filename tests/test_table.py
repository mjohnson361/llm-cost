from llm_cost.table import format_int, format_money, render_table


def test_format_int_groups_thousands():
    assert format_int(0) == "0"
    assert format_int(12000) == "12,000"
    assert format_int(104000) == "104,000"


def test_format_money_uses_four_decimals():
    assert format_money(0.06) == "$0.0600"
    assert format_money(0.0) == "$0.0000"
    assert format_money(0.56) == "$0.5600"


def test_format_money_groups_thousands():
    assert format_money(12345.6) == "$12,345.6000"


def test_render_table_matches_reference_layout():
    # Column widths below (12, 6, 5, 7) are the widest cell in each column of
    # this fixture, matching the layout shown in the README.
    headers = ["item", "tokens", "$/1M", "cost"]
    rows = [
        ["input", "12,000", "5.00", "$0.0600"],
        ["cached input", "0", "0.50", "$0.0000"],
        ["cache write", "0", "6.25", "$0.0000"],
        ["output", "800", "25.00", "$0.0200"],
    ]
    widths = (12, 6, 5, 7)

    def row(cells):
        first, rest = cells[0], cells[1:]
        return "  ".join(
            [first.ljust(widths[0])]
            + [cell.rjust(width) for cell, width in zip(rest, widths[1:])]
        )

    expected = "\n".join(
        [row(headers), "  ".join("-" * width for width in widths)]
        + [row(data_row) for data_row in rows]
    )
    assert render_table(headers, rows) == expected
    assert expected.split("\n")[0] == "item          tokens   $/1M     cost"
    assert expected.split("\n")[1] == "------------  ------  -----  -------"


def test_render_table_first_column_left_justified_others_right_justified():
    rendered = render_table(["model", "calls"], [["claude-opus-5", "2"], ["x", "10"]])
    lines = rendered.split("\n")
    assert lines[2].startswith("claude-opus-5")
    assert lines[3].startswith("x   ")


def test_render_table_widens_columns_to_fit_data_not_just_header():
    rendered = render_table(["model"], [["a-very-long-model-name"]])
    header_line, separator_line, row_line = rendered.split("\n")
    assert len(header_line) == len("a-very-long-model-name")
    assert separator_line == "-" * len("a-very-long-model-name")
