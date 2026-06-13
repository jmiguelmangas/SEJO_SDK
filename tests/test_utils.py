import json
import logging
import sys
from types import ModuleType

import pytest

from SEJO_SDK.utils import Utils
from SEJO_SDK.utils.postgresql_connector import PostgresqlConnector


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def error(self, message, exc_info=False):
        self.messages.append(("error", message, exc_info))

    def warning(self, message):
        self.messages.append(("warning", message))

    def debug(self, message):
        self.messages.append(("debug", message))


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False
        self.executed = None

    def execute(self, query):
        self.executed = query

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_utils_load_save_json_and_env(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "data.json"

    Utils.save_to_json({"ok": True}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert Utils.load_json_file(path) == {"ok": True}

    monkeypatch.setenv("SEJO_TEST_VALUE", "yes")

    assert Utils.get_env_variable("SEJO_TEST_VALUE") == "yes"
    assert Utils.get_env_variable("MISSING", default="fallback") == "fallback"


def test_utils_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Utils.load_json_file(tmp_path / "missing.json")


def test_utils_get_env_variable_without_default_raises(monkeypatch):
    monkeypatch.delenv("SEJO_MISSING", raising=False)

    with pytest.raises(ValueError):
        Utils.get_env_variable("SEJO_MISSING")


def test_utils_logging_delegates_to_logger():
    logger = FakeLogger()
    utils = Utils(logger=logger)

    utils.log_info("info")
    utils.log_error("error", exc_info=True)
    utils.log_warning("warning")
    utils.log_debug("debug")

    assert logger.messages == [
        ("info", "info"),
        ("error", "error", True),
        ("warning", "warning"),
        ("debug", "debug"),
    ]


def test_default_logger_is_configured():
    logger = Utils._setup_default_logger()

    assert isinstance(logger, logging.Logger)


def test_postgresql_connector_execute_query_and_disconnect():
    cursor = FakeCursor(rows=[{"answer": 42}])
    connection = FakeConnection(cursor)
    connector = PostgresqlConnector(
        host="localhost",
        port=5432,
        user="user",
        password="password",
        database="db",
    )
    connector.connection = connection

    assert connector.execute_query("select 42") == [{"answer": 42}]
    assert cursor.executed == "select 42"
    assert cursor.closed is True

    connector.disconnect()

    assert connection.closed is True


def test_postgresql_connector_connect_uses_psycopg2(monkeypatch):
    calls = []
    cursor_marker = object()

    fake_psycopg2 = ModuleType("psycopg2")

    def connect(**kwargs):
        calls.append(kwargs)
        connection = FakeConnection(FakeCursor(rows=[]))
        connection.autocommit = False
        return connection

    fake_psycopg2.connect = connect
    fake_extras = ModuleType("psycopg2.extras")
    fake_extras.RealDictCursor = cursor_marker

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)

    connector = PostgresqlConnector(
        host="localhost",
        port=5432,
        user="user",
        password="password",
        database="db",
    )

    connector.connect()

    assert connector.connection.autocommit is True
    assert calls == [
        {
            "host": "localhost",
            "port": 5432,
            "user": "user",
            "password": "password",
            "database": "db",
            "cursor_factory": cursor_marker,
        }
    ]


def test_postgresql_connector_connect_wraps_connection_errors(monkeypatch):
    fake_psycopg2 = ModuleType("psycopg2")

    def connect(**kwargs):
        raise OSError("network down")

    fake_psycopg2.connect = connect
    fake_extras = ModuleType("psycopg2.extras")
    fake_extras.RealDictCursor = object()

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)

    connector = PostgresqlConnector(
        host="localhost",
        port=5432,
        user="user",
        password="password",
        database="db",
    )

    with pytest.raises(RuntimeError, match="Error connecting to PostgreSQL"):
        connector.connect()


def test_postgresql_connector_execute_query_wraps_query_errors():
    class FailingCursor(FakeCursor):
        def execute(self, query):
            raise ValueError("bad query")

    cursor = FailingCursor(rows=[])
    connector = PostgresqlConnector(
        host="localhost",
        port=5432,
        user="user",
        password="password",
        database="db",
    )
    connector.connection = FakeConnection(cursor)

    with pytest.raises(RuntimeError, match="Error executing query"):
        connector.execute_query("select nope")

    assert cursor.closed is True


def test_postgresql_connector_execute_requires_connection():
    connector = PostgresqlConnector(
        host="localhost",
        port=5432,
        user="user",
        password="password",
        database="db",
    )

    with pytest.raises(RuntimeError, match="has not been opened"):
        connector.execute_query("select 42")
