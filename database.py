import sqlite3
from datetime import datetime


def create_database():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    # Create table if it does not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            disease TEXT,
            confidence REAL,
            date_time TEXT
        )
    """)

    # Add date_time column to old database
    try:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN date_time TEXT"
        )
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def save_prediction(image_name, disease, confidence):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    date_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    cursor.execute("""
        INSERT INTO predictions
        (image_name, disease, confidence, date_time)
        VALUES (?, ?, ?, ?)
    """, (image_name, disease, confidence, date_time))

    conn.commit()
    conn.close()