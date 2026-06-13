"""PostgreSQL utility connector."""

from typing import Any

from SEJO_SDK.errors import ProviderDependencyError


class PostgresqlConnector:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self) -> None:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as exc:
            raise ProviderDependencyError(
                "Install PostgreSQL support with `pip install sejo-sdk[postgres]`."
            ) from exc

        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursor_factory=RealDictCursor,
            )
            self.connection.autocommit = True
        except Exception as exc:
            raise RuntimeError(f"Error connecting to PostgreSQL: {str(exc)}") from exc

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        if self.connection is None:
            raise RuntimeError("PostgreSQL connection has not been opened.")

        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise RuntimeError(f"Error executing query: {str(exc)}") from exc
        finally:
            if cursor is not None:
                cursor.close()
