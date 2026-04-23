"""DuckDB storage utilities for StAR-E."""

from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

from star_e.config import settings


def get_db_path() -> Path:
    """Get path to DuckDB database file."""
    return settings.data_dir / "star_e.duckdb"


def get_connection(db_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def save_to_duckdb(
    df: pl.DataFrame,
    table_name: str,
    db_path: Optional[Path] = None,
    if_exists: str = "replace",
) -> None:
    """
    Save Polars DataFrame to DuckDB table.

    Args:
        df: DataFrame to save
        table_name: Name of the table
        db_path: Path to database file
        if_exists: How to handle existing table ("replace", "append", "fail")
    """
    conn = get_connection(db_path)

    if if_exists == "replace":
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    elif if_exists == "append":
        try:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        except duckdb.CatalogException:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    elif if_exists == "fail":
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    else:
        raise ValueError(f"Unknown if_exists value: {if_exists}")

    conn.close()


def load_from_duckdb(
    table_name: str,
    db_path: Optional[Path] = None,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load data from DuckDB table to Polars DataFrame.

    Args:
        table_name: Name of the table
        db_path: Path to database file
        columns: Columns to select (None = all)
        where: WHERE clause for filtering

    Returns:
        Polars DataFrame with query results
    """
    conn = get_connection(db_path)

    cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols} FROM {table_name}"
    if where:
        query += f" WHERE {where}"

    result = conn.execute(query).pl()
    conn.close()

    return result


def query_duckdb(
    query: str,
    db_path: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Execute arbitrary SQL query on DuckDB.

    Args:
        query: SQL query string
        db_path: Path to database file

    Returns:
        Polars DataFrame with query results
    """
    conn = get_connection(db_path)
    result = conn.execute(query).pl()
    conn.close()
    return result


def list_tables(db_path: Optional[Path] = None) -> list[str]:
    """List all tables in database."""
    conn = get_connection(db_path)
    result = conn.execute("SHOW TABLES").fetchall()
    conn.close()
    return [row[0] for row in result]


def table_info(
    table_name: str,
    db_path: Optional[Path] = None,
) -> dict:
    """Get information about a table."""
    conn = get_connection(db_path)

    schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    conn.close()

    return {
        "name": table_name,
        "columns": [{"name": row[0], "type": row[1]} for row in schema],
        "row_count": row_count,
    }
