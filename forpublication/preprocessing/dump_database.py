"""
Dump the entire `mimic_thesis` Postgres database to local files.
=================================================================
Run once. Afterwards, experiments can read these files instead of needing a
running Postgres server (and the loaders can drop psycopg + DB_CONFIG entirely).

Auto-discovers EVERY table in the `public` schema — no hardcoded schema, so it
dumps whatever is in the database.

⚠️  The output contains patient data (MIMIC / PhysioNet). Gitignore the output
    directory; never push it publicly. The password below is local-only —
    move it to an env var / secret before sharing this file.

Usage:
    python dump_database.py
"""

from pathlib import Path

import pandas as pd
import psycopg

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimic_thesis",
    "user": "thesis_user",
    "password": "thesis",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "db" / "mimic_data"
FORMAT = "parquet"   # "parquet" (recommended — small + typed) or "csv"


def get_connection():
    return psycopg.connect(**DB_CONFIG)


def list_tables(conn) -> list[str]:
    """Every base table in the public schema."""
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    return pd.read_sql(query, conn)["table_name"].tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()

    tables = list_tables(conn)
    print(f"Found {len(tables)} table(s): {tables}")
    print(f"Writing {FORMAT} files to: {OUT_DIR}\n")

    for table in tables:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)

        if FORMAT == "parquet":
            out_path = OUT_DIR / f"{table}.parquet"
            df.to_parquet(out_path, index=False)
        elif FORMAT == "csv":
            out_path = OUT_DIR / f"{table}.csv"
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unknown FORMAT: {FORMAT!r} (use 'parquet' or 'csv')")

        size_mb = out_path.stat().st_size / 1e6
        print(f"  {table:<20} {len(df):>8,} rows  ->  {out_path.name} ({size_mb:.1f} MB)")

    conn.close()
    print(f"\nDone. {len(tables)} table(s) dumped to {OUT_DIR}")


if __name__ == "__main__":
    main()
