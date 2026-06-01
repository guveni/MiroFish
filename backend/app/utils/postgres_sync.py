"""
Dynamic SQLite-to-PostgreSQL synchronization utility.
Ensures virtual platform data is safely persisted in the central high-throughput PostgreSQL
database, supporting multi-user isolation and robust scalability.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from ..database import db

logger = logging.getLogger(__name__)


def get_postgres_connection():
    """Returns a direct connection to the PostgreSQL engine."""
    if db.engine is None:
        # If db is not yet initialized (e.g. running in standalone scripts), initialize it now.
        from ..config import Config
        db.init_app(Config.SQLALCHEMY_DATABASE_URI)
    return db.engine.connect()


def sync_sqlite_to_postgres(sqlite_db_path: str, simulation_id: str, platform: str):
    """
    Dynamically copy/upsert all tables from an SQLite simulation database to PostgreSQL.
    Creates tables dynamically as 'sim_platform_{table_name}' to avoid collisions,
    adding 'simulation_id' and 'platform' columns for perfect multi-user isolation.
    """
    if not os.path.exists(sqlite_db_path):
        logger.warning(f"SQLite simulation database not found at: {sqlite_db_path}. Skipping sync.")
        return

    logger.info(f"Syncing SQLite database {sqlite_db_path} to PostgreSQL for simulation {simulation_id} ({platform})...")

    # Connect to local SQLite in read-only and no-lock mode to prevent locking, SHM mapping,
    # or journal creation issues when the simulation process is actively writing,
    # especially on Docker bind mounts (e.g., VirtioFS on macOS).
    db_abs_path = os.path.abspath(sqlite_db_path)
    # Prefer immutable=1 for read-only sync on WAL databases over Docker bind mounts
    # (macOS VirtioFS): mode=ro can still try to create/open -wal/-shm sidecars and fail.
    sqlite_conn = None
    tmp_path = None
    connect_errors = []
    for uri in (
        f"file:{db_abs_path}?mode=ro&immutable=1",
        f"file:{db_abs_path}?mode=ro",
        f"file:{db_abs_path}?mode=ro&nolock=1",
    ):
        try:
            candidate = sqlite3.connect(uri, uri=True, timeout=30.0)
            candidate.execute("SELECT name FROM sqlite_master LIMIT 1")
            sqlite_conn = candidate
            break
        except sqlite3.OperationalError as e:
            connect_errors.append(f"{uri}: {e}")
            continue

    if sqlite_conn is None:
        import shutil
        import tempfile

        logger.warning(
            "SQLite read-only open failed (%s). Copying to temp file for sync.",
            "; ".join(connect_errors),
        )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = tmp.name
            shutil.copy2(db_abs_path, tmp_path)
            for suffix in ("-wal", "-shm"):
                sidecar = db_abs_path + suffix
                if os.path.exists(sidecar):
                    shutil.copy2(sidecar, tmp_path + suffix)
            sqlite_conn = sqlite3.connect(tmp_path, timeout=30.0)
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.error(f"Unable to open SQLite database for sync: {e}")
            return

    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Get all tables in SQLite
    sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row['name'] for row in sqlite_cursor.fetchall()]

    pg_conn = get_postgres_connection()
    transaction = pg_conn.begin()

    try:
        for table in tables:
            # We prefix the PostgreSQL table to organize nicely
            pg_table_name = f"sim_{table}"

            # 1. Fetch SQLite schema for this table
            sqlite_cursor.execute(f"PRAGMA table_info({table})")
            columns_info = sqlite_cursor.fetchall()

            # Map SQLite types to PostgreSQL types
            # SQLite types: INT, INTEGER, TINYINT, SMALLINT, MEDIUMINT, BIGINT, UNSIGNED BIG INT, INT2, INT8,
            # CHARACTER, VARCHAR, VARYING CHARACTER, NCHAR, NATIVE CHARACTER, TEXT, CLOB, BLOB, REAL, DOUBLE,
            # DOUBLE PRECISION, FLOAT, NUMERIC, DECIMAL, DATE, DATETIME, BOOLEAN
            pg_cols_def = []
            col_names = []
            
            for col in columns_info:
                name = col['name']
                col_type = col['type'].upper()
                is_nullable = "NULL" if col['notnull'] == 0 else "NOT NULL"
                has_pk = "PRIMARY KEY" if col['pk'] == 1 else ""

                col_names.append(name)

                # Simple type mapper
                if "INT" in col_type:
                    pg_type = "BIGINT" if "BIG" in col_type or name == "id" else "INTEGER"
                elif "CHAR" in col_type or "TEXT" in col_type or "CLOB" in col_type:
                    pg_type = "TEXT"
                elif "REAL" in col_type or "DOUBLE" in col_type or "FLOAT" in col_type:
                    pg_type = "DOUBLE PRECISION"
                elif "BOOL" in col_type:
                    pg_type = "BOOLEAN"
                else:
                    pg_type = "TEXT"  # Fallback

                # Primary keys in SQLite might be autoincrement, we drop PRIMARY KEY constraint on the column
                # and instead use composite primary key of (simulation_id, id/rowid) or a separate unique constraint.
                pg_cols_def.append(f'"{name}" {pg_type} {is_nullable}')

            # Drop exact primary keys to support multi-simulation rows in the same table.
            # We define primary key as (simulation_id, platform, id) or whatever SQLite primary key was.
            sqlite_pks = [col['name'] for col in columns_info if col['pk'] == 1]
            if not sqlite_pks:
                # Fallback pk
                sqlite_pks = ["id"] if "id" in col_names else []

            pk_def = ""
            if sqlite_pks:
                pk_cols = ", ".join(f'"{pk}"' for pk in sqlite_pks)
                pk_def = f', PRIMARY KEY ("simulation_id", "platform", {pk_cols})'

            # 2. Create PostgreSQL table if it doesn't exist
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS "{pg_table_name}" (
                    "simulation_id" TEXT NOT NULL,
                    "platform" TEXT NOT NULL,
                    {", ".join(pg_cols_def)}
                    {pk_def}
                )
            """
            pg_conn.execute(text(create_table_sql))

            # 3. Read data from SQLite
            sqlite_cursor.execute(f"SELECT * FROM {table}")
            rows = sqlite_cursor.fetchall()

            if not rows:
                continue

            # 4. Upsert/Insert into PostgreSQL
            # We build an INSERT ... ON CONFLICT ("simulation_id", "platform", pk) DO UPDATE set ...
            placeholders = ", ".join([f":{col}" for col in col_names])
            all_cols = ['"simulation_id"', '"platform"'] + [f'"{col}"' for col in col_names]
            all_placeholders = [":simulation_id", ":platform"] + [f":{col}" for col in col_names]

            upsert_sql = f"""
                INSERT INTO "{pg_table_name}" ({", ".join(all_cols)})
                VALUES ({", ".join(all_placeholders)})
            """
            
            if sqlite_pks:
                conflict_cols = f'"simulation_id", "platform", ' + ", ".join(f'"{pk}"' for pk in sqlite_pks)
                update_assignments = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in col_names if col not in sqlite_pks])
                if update_assignments:
                    upsert_sql += f"""
                        ON CONFLICT ({conflict_cols})
                        DO UPDATE SET {update_assignments}
                    """
                else:
                    upsert_sql += f"""
                        ON CONFLICT ({conflict_cols})
                        DO NOTHING
                    """
            else:
                # No PK, just simple insert or custom clear first
                # For non-pk tables, we can delete existing for this simulation_id and insert fresh
                delete_sql = f'DELETE FROM "{pg_table_name}" WHERE "simulation_id" = :simulation_id AND "platform" = :platform'
                pg_conn.execute(text(delete_sql), {"simulation_id": simulation_id, "platform": platform})

            # Batch insert rows
            params_batch = []
            for row in rows:
                param_dict = {"simulation_id": simulation_id, "platform": platform}
                for col in col_names:
                    val = row[col]
                    # Convert dicts/lists to JSON string for Postgres compatibility
                    if isinstance(val, (dict, list)):
                        val = json.dumps(val)
                    param_dict[col] = val
                params_batch.append(param_dict)

            # Execute batch insert/upsert
            pg_conn.execute(text(upsert_sql), params_batch)

        transaction.commit()
        logger.info(f"Successfully synchronized SQLite database for simulation {simulation_id} to PostgreSQL.")
    except Exception as e:
        transaction.rollback()
        logger.error(f"Failed to synchronize SQLite to PostgreSQL: {e}", exc_info=True)
    finally:
        pg_conn.close()
        sqlite_conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            for suffix in ("-wal", "-shm"):
                sidecar = tmp_path + suffix
                if os.path.exists(sidecar):
                    os.unlink(sidecar)


def has_table_in_pg(table_name: str) -> bool:
    """Helper to check if a specific table exists in PostgreSQL."""
    pg_conn = get_postgres_connection()
    try:
        # Check standard postgres information_schema
        sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            )
        """
        result = pg_conn.execute(text(sql), {"table_name": table_name}).scalar()
        return bool(result)
    except Exception as e:
        logger.warning(f"Failed to check table existence for {table_name}: {e}")
        return False
    finally:
        pg_conn.close()


def get_simulation_posts_from_pg(simulation_id: str, platform: str, limit: int, offset: int) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Retrieve posts from PostgreSQL for perfect scalability."""
    if not has_table_in_pg("sim_post"):
        return None

    pg_conn = get_postgres_connection()
    try:
        # Fetch posts
        sql_fetch = """
            SELECT * FROM sim_post 
            WHERE simulation_id = :simulation_id AND platform = :platform
            ORDER BY created_at DESC 
            LIMIT :limit OFFSET :offset
        """
        res_fetch = pg_conn.execute(text(sql_fetch), {
            "simulation_id": simulation_id,
            "platform": platform,
            "limit": limit,
            "offset": offset
        })
        
        # Convert to dict list
        keys = res_fetch.keys()
        posts = []
        for row in res_fetch.fetchall():
            post_dict = {}
            for k, val in zip(keys, row):
                if k not in ("simulation_id", "platform"):
                    # Unpack JSON strings if applicable
                    if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                        try:
                            val = json.loads(val)
                        except Exception:
                            pass
                    post_dict[k] = val
            posts.append(post_dict)

        # Count posts
        sql_count = "SELECT COUNT(*) FROM sim_post WHERE simulation_id = :simulation_id AND platform = :platform"
        total = pg_conn.execute(text(sql_count), {
            "simulation_id": simulation_id,
            "platform": platform
        }).scalar()

        return posts, int(total) if total is not None else 0
    except Exception as e:
        logger.error(f"Failed to fetch posts from PostgreSQL: {e}")
        return None
    finally:
        pg_conn.close()


def get_simulation_comments_from_pg(simulation_id: str, post_id: Optional[str], limit: int, offset: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieve comments from PostgreSQL for perfect scalability."""
    if not has_table_in_pg("sim_comment"):
        return None

    pg_conn = get_postgres_connection()
    try:
        # Fetch comments
        if post_id:
            sql_fetch = """
                SELECT * FROM sim_comment 
                WHERE simulation_id = :simulation_id AND post_id = :post_id
                ORDER BY created_at DESC 
                LIMIT :limit OFFSET :offset
            """
            params = {"simulation_id": simulation_id, "post_id": post_id, "limit": limit, "offset": offset}
        else:
            sql_fetch = """
                SELECT * FROM sim_comment 
                WHERE simulation_id = :simulation_id
                ORDER BY created_at DESC 
                LIMIT :limit OFFSET :offset
            """
            params = {"simulation_id": simulation_id, "limit": limit, "offset": offset}

        res_fetch = pg_conn.execute(text(sql_fetch), params)
        keys = res_fetch.keys()
        comments = []
        for row in res_fetch.fetchall():
            comment_dict = {}
            for k, val in zip(keys, row):
                if k not in ("simulation_id", "platform"):
                    if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                        try:
                            val = json.loads(val)
                        except Exception:
                            pass
                    comment_dict[k] = val
            comments.append(comment_dict)

        return comments
    except Exception as e:
        logger.error(f"Failed to fetch comments from PostgreSQL: {e}")
        return None
    finally:
        pg_conn.close()


def get_interview_history_from_pg(simulation_id: str, platform: str, agent_id: Optional[int], limit: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieve interview traces from PostgreSQL for perfect scalability."""
    if not has_table_in_pg("sim_trace"):
        return None

    pg_conn = get_postgres_connection()
    try:
        if agent_id is not None:
            sql = """
                SELECT user_id, info, created_at
                FROM sim_trace
                WHERE simulation_id = :simulation_id AND platform = :platform AND action = 'interview' AND user_id = :agent_id
                ORDER BY created_at DESC
                LIMIT :limit
            """
            params = {"simulation_id": simulation_id, "platform": platform, "agent_id": agent_id, "limit": limit}
        else:
            sql = """
                SELECT user_id, info, created_at
                FROM sim_trace
                WHERE simulation_id = :simulation_id AND platform = :platform AND action = 'interview'
                ORDER BY created_at DESC
                LIMIT :limit
            """
            params = {"simulation_id": simulation_id, "platform": platform, "limit": limit}

        res = pg_conn.execute(text(sql), params)
        results = []
        for user_id, info_json, created_at in res.fetchall():
            info = {}
            if info_json:
                if isinstance(info_json, str):
                    try:
                        info = json.loads(info_json)
                    except json.JSONDecodeError:
                        info = {"raw": info_json}
                elif isinstance(info_json, dict):
                    info = info_json

            results.append({
                "agent_id": user_id,
                "response": info.get("response", info),
                "prompt": info.get("prompt", ""),
                "timestamp": created_at,
                "platform": platform
            })
        return results
    except Exception as e:
        logger.error(f"Failed to fetch interview traces from PostgreSQL: {e}")
        return None
    finally:
        pg_conn.close()

