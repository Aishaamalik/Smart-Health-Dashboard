import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'health.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS health_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        timestamp TEXT,
        sensor_id INTEGER,
        sensor_type TEXT,
        temperature REAL,
        systolic_bp REAL,
        diastolic_bp REAL,
        heart_rate REAL,
        device_battery_level INTEGER,
        target_blood_pressure REAL,
        target_heart_rate REAL,
        target_health_status TEXT,
        battery_level INTEGER,
        sleep REAL,
        steps INTEGER
    )''')
    conn.commit()
    conn.close()

def insert_records(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('health_records', conn, if_exists='append', index=False)
    conn.close()

def fetch_all_records():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM health_records', conn)
    conn.close()
    return df 