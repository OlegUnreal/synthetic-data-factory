import json

from factory.logging_config import JsonFormatter
import logging


def test_json_formatter():
    fmt = JsonFormatter()
    rec = logging.LogRecord(name="t", level=logging.INFO, pathname="", lineno=0, msg="hi %s", args=("x",), exc_info=None)
    data = json.loads(fmt.format(rec))
    assert data["msg"] == "hi x"
    assert "ts" in data
