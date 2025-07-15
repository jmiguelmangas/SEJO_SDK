import psycopg2
from typing import List, Dict, Any


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
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.connection.autocommit = True
        except Exception as e:
            raise Exception(f"Error connecting to PostgreSQL: {str(e)}")

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
        finally:
            cursor.close()  

            