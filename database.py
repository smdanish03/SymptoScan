"""
Database abstraction layer for SymptoScan AI.
Supports SQLite (zero-config, default for Render & local development),
PostgreSQL (Render PostgreSQL via DATABASE_URL), and MySQL.
Auto-initializes tables on boot.
"""

import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(INSTANCE_DIR, "symptoscan.db")

def _get_db_type():
    db_engine = os.getenv("DB_ENGINE", "").lower().strip()
    database_url = os.getenv("DATABASE_URL", "")
    
    if db_engine == "mysql" or (os.getenv("DB_HOST") and not database_url):
        # Check if mysql-connector is usable
        try:
            import mysql.connector
            return "mysql"
        except ImportError:
            pass

    if database_url.startswith("postgres://") or database_url.startswith("postgresql://"):
        try:
            import psycopg2
            return "postgres"
        except ImportError:
            pass

    return "sqlite"

DB_TYPE = _get_db_type()

def get_connection():
    """Returns an active database connection."""
    global DB_TYPE
    
    if DB_TYPE == "mysql":
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST", "localhost"),
                user=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "symptoscan"),
                connection_timeout=5
            )
            return conn, "mysql"
        except Exception as e:
            print(f"[Database] MySQL connection failed ({e}). Falling back to SQLite.")
            DB_TYPE = "sqlite"

    if DB_TYPE == "postgres":
        try:
            import psycopg2
            import psycopg2.extras
            db_url = os.getenv("DATABASE_URL")
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            conn = psycopg2.connect(db_url)
            return conn, "postgres"
        except Exception as e:
            print(f"[Database] PostgreSQL connection failed ({e}). Falling back to SQLite.")
            DB_TYPE = "sqlite"

    # SQLite (default, rock-solid, zero-config)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def init_db():
    """Creates database tables automatically if they do not exist."""
    conn, db_type = get_connection()
    cursor = conn.cursor()

    try:
        if db_type == "sqlite":
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                severity TEXT DEFAULT 'Moderate',
                specialist TEXT DEFAULT 'General Physician',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prediction_id INTEGER,
                rating INTEGER DEFAULT 5,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

        elif db_type == "postgres":
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                name VARCHAR(255) NOT NULL,
                age INTEGER NOT NULL,
                gender VARCHAR(50) NOT NULL,
                symptoms TEXT NOT NULL,
                disease VARCHAR(255) NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                severity VARCHAR(50) DEFAULT 'Moderate',
                specialist VARCHAR(255) DEFAULT 'General Physician',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                prediction_id INTEGER,
                rating INTEGER DEFAULT 5,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

        elif db_type == "mysql":
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                name VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                gender VARCHAR(50) NOT NULL,
                symptoms TEXT NOT NULL,
                disease VARCHAR(255) NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                severity VARCHAR(50) DEFAULT 'Moderate',
                specialist VARCHAR(255) DEFAULT 'General Physician',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                prediction_id INT,
                rating INT DEFAULT 5,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

        # Automatic column migration for existing tables
        try:
            if db_type == "mysql":
                cursor.execute("SHOW COLUMNS FROM predictions")
                existing_cols = [row[0] for row in cursor.fetchall()]
                if "confidence" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN confidence FLOAT DEFAULT 0.0")
                if "severity" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN severity VARCHAR(50) DEFAULT 'Moderate'")
                if "specialist" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN specialist VARCHAR(255) DEFAULT 'General Physician'")
                if "notes" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN notes TEXT")
            elif db_type == "sqlite":
                cursor.execute("PRAGMA table_info(predictions)")
                existing_cols = [row[1] for row in cursor.fetchall()]
                if "confidence" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN confidence REAL DEFAULT 0.0")
                if "severity" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN severity TEXT DEFAULT 'Moderate'")
                if "specialist" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN specialist TEXT DEFAULT 'General Physician'")
                if "notes" not in existing_cols:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN notes TEXT")
            elif db_type == "postgres":
                cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.0;")
                cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS severity VARCHAR(50) DEFAULT 'Moderate';")
                cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS specialist VARCHAR(255) DEFAULT 'General Physician';")
                cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS notes TEXT;")
        except Exception as mig_err:
            print(f"[Database] Migration notice: {mig_err}")

        conn.commit()
        print(f"[Database] Tables initialized successfully using {db_type.upper()}.")
    finally:
        cursor.close()
        conn.close()

def execute_query(sql, params=()):
    """Executes an INSERT, UPDATE, or DELETE query and returns the last inserted row id."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    last_id = None

    try:
        if db_type == "sqlite":
            formatted_sql = sql.replace("%s", "?")
            cursor.execute(formatted_sql, params)
            last_id = cursor.lastrowid
        else:
            cursor.execute(sql, params)
            if db_type == "mysql":
                last_id = cursor.lastrowid
            elif db_type == "postgres":
                try:
                    last_id = cursor.fetchone()[0]
                except Exception:
                    pass

        conn.commit()
        return last_id
    finally:
        cursor.close()
        conn.close()

def query_one(sql, params=()):
    """Executes a SELECT query and returns a single dictionary row or None."""
    conn, db_type = get_connection()
    try:
        if db_type == "sqlite":
            formatted_sql = sql.replace("%s", "?")
            cursor = conn.cursor()
            cursor.execute(formatted_sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        elif db_type == "postgres":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        elif db_type == "mysql":
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return row
    finally:
        conn.close()

def query_all(sql, params=()):
    """Executes a SELECT query and returns a list of dictionary rows."""
    conn, db_type = get_connection()
    try:
        if db_type == "sqlite":
            formatted_sql = sql.replace("%s", "?")
            cursor = conn.cursor()
            cursor.execute(formatted_sql, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        elif db_type == "postgres":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        elif db_type == "mysql":
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()
