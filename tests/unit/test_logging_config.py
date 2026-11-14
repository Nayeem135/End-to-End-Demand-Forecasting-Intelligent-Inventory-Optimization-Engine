import json
import logging

import pytest

from demand_engine.logging_config import JsonFormatter, configure_logging


def test_configure_logging_sets_level_and_quiets_noisy_libs() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("lightgbm").level == logging.WARNING
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_text_format_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "text")
    logging.getLogger("demo.text").info("hello %s", "world")
    err = capsys.readouterr().err
    assert "| INFO    | demo.text | hello world" in err


def test_json_format_is_one_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logging.getLogger("demo.json").warning("stock %d units", 5)
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("demo.json").exception("failed")
    configure_logging("INFO")  # restore default for other tests
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2  # the traceback is escaped inside its JSON line
    first = json.loads(lines[0])
    assert first["level"] == "WARNING"
    assert first["logger"] == "demo.json"
    assert first["message"] == "stock 5 units"
    assert first["time"].endswith("+00:00")
    error = json.loads(lines[1])
    assert error["message"] == "failed"
    assert "ValueError: boom" in error["exception"]


def test_json_formatter_escapes_newlines() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "line1\nline2", None, None)
    out = JsonFormatter().format(record)
    assert "\n" not in out
    assert json.loads(out)["message"] == "line1\nline2"
