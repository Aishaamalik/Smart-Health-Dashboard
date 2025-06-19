from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import database
import time
from fastapi.responses import JSONResponse

database.init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../healthcare_iot_target_dataset.csv')
CACHE_TTL = 300  # 5 minutes
cache = {
    "stats": {"data": None, "timestamp": 0},
    "predict": {"data": None, "timestamp": 0},
    "alerts": {"data": None, "timestamp": 0},
}

def is_cache_valid(key):
    return cache[key]["data"] is not None and (time.time() - cache[key]["timestamp"] < CACHE_TTL)

# Helper to get data from DB or CSV

def get_data():
    try:
        df = database.fetch_all_records()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception:
        pass
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df = df.dropna(subset=["Timestamp", "Heart_Rate (bpm)", "Temperature (°C)"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.rename(columns={
        "Patient_ID": "patient_id",
        "Timestamp": "timestamp",
        "Sensor_ID": "sensor_id",
        "Sensor_Type": "sensor_type",
        "Temperature (°C)": "temperature",
        "Systolic_BP (mmHg)": "systolic_bp",
        "Diastolic_BP (mmHg)": "diastolic_bp",
        "Heart_Rate (bpm)": "heart_rate",
        "Device_Battery_Level (%)": "device_battery_level",
        "Target_Blood_Pressure": "target_blood_pressure",
        "Target_Heart_Rate": "target_heart_rate",
        "Target_Health_Status": "target_health_status",
        "Battery_Level (%)": "battery_level"
    })
    return df

@app.post("/api/upload")
def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    df = pd.read_csv(file.file)
    # Validate columns
    required_cols = [
        "Patient_ID", "Timestamp", "Sensor_ID", "Sensor_Type", "Temperature (°C)",
        "Systolic_BP (mmHg)", "Diastolic_BP (mmHg)", "Heart_Rate (bpm)", "Device_Battery_Level (%)",
        "Target_Blood_Pressure", "Target_Heart_Rate", "Target_Health_Status", "Battery_Level (%)"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")
    # Clean and insert
    df = df.drop_duplicates()
    df = df.dropna(subset=["Timestamp", "Heart_Rate (bpm)", "Temperature (°C)"])
    # Rename columns to match DB
    df.columns = [
        "patient_id", "timestamp", "sensor_id", "sensor_type", "temperature", "systolic_bp", "diastolic_bp",
        "heart_rate", "device_battery_level", "target_blood_pressure", "target_heart_rate", "target_health_status", "battery_level"
    ]
    database.insert_records(df)
    # Invalidate cache
    for key in cache:
        cache[key]["data"] = None
        cache[key]["timestamp"] = 0
    return {"message": "Data uploaded and saved."}

@app.get("/api/stats")
def get_stats():
    if is_cache_valid("stats"):
        return cache["stats"]["data"]
    df = get_data()
    avg_heart_rate = df["heart_rate"].mean()
    avg_temp = df["temperature"].mean()
    avg_sys = df["systolic_bp"].mean()
    avg_dia = df["diastolic_bp"].mean()
    result = {
        "average_heart_rate": round(avg_heart_rate, 2),
        "average_temperature": round(avg_temp, 2),
        "average_systolic_bp": round(avg_sys, 2),
        "average_diastolic_bp": round(avg_dia, 2),
    }
    cache["stats"]["data"] = result
    cache["stats"]["timestamp"] = time.time()
    return result

@app.get("/api/predict")
def get_prediction():
    if is_cache_valid("predict"):
        return cache["predict"]["data"]
    df = get_data()
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)
    df["ordinal_time"] = df["timestamp"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") if isinstance(x, str) else x)
    df["ordinal_time"] = df["ordinal_time"].map(lambda x: x.toordinal())
    X = df["ordinal_time"].values.reshape(-1, 1)
    y = df["heart_rate"].values
    model = LinearRegression()
    model.fit(X, y)
    last_date = df["timestamp"].max()
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
    preds = []
    for i in range(1, 8):
        future_date = last_date + timedelta(days=i)
        pred = model.predict(np.array([[future_date.toordinal()]]))[0]
        preds.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "predicted_heart_rate": round(pred, 2)
        })
    result = {"predictions": preds}
    cache["predict"]["data"] = result
    cache["predict"]["timestamp"] = time.time()
    return result

@app.get("/api/alerts")
def get_alerts():
    if is_cache_valid("alerts"):
        return cache["alerts"]["data"]
    df = get_data()
    alerts = []
    # Abnormal heart rate
    abnormal_hr = df[(df["heart_rate"] < 50) | (df["heart_rate"] > 120)]
    for _, row in abnormal_hr.iterrows():
        alerts.append({
            "timestamp": row["timestamp"],
            "type": "Heart Rate",
            "value": row["heart_rate"],
            "message": "Abnormal heart rate detected"
        })
    # Abnormal temperature
    abnormal_temp = df[(df["temperature"] < 35.5) | (df["temperature"] > 38.5)]
    for _, row in abnormal_temp.iterrows():
        alerts.append({
            "timestamp": row["timestamp"],
            "type": "Temperature",
            "value": row["temperature"],
            "message": "Abnormal temperature detected"
        })
    # Advanced spike detection (rolling window)
    df = df.sort_values("timestamp")
    for col, label in [("heart_rate", "Heart Rate Spike"), ("temperature", "Temperature Spike")]:
        rolling_mean = df[col].rolling(window=5, min_periods=3).mean()
        rolling_std = df[col].rolling(window=5, min_periods=3).std()
        spikes = (abs(df[col] - rolling_mean) > 2 * rolling_std)
        for idx in df[spikes].index:
            alerts.append({
                "timestamp": df.loc[idx, "timestamp"],
                "type": label,
                "value": df.loc[idx, col],
                "message": f"Sudden spike/drop detected in {col.replace('_', ' ')}"
            })
    result = {"alerts": alerts}
    cache["alerts"]["data"] = result
    cache["alerts"]["timestamp"] = time.time()
    return result

@app.get("/api/data")
def get_full_data():
    """Return the cleaned dataset as JSON (for frontend graphing)."""
    df = get_data()
    # Convert timestamps to string for JSON serialization
    df["timestamp"] = df["timestamp"].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/api/column_stats")
def get_column_stats():
    """Return detailed stats (mean, median, mode, min, max, std) for each numeric column."""
    df = get_data()
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = float(df[col].mean()) if not pd.isnull(df[col].mean()) else None
        col_median = float(df[col].median()) if not pd.isnull(df[col].median()) else None
        col_mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
        if hasattr(col_mode, 'item'):
            col_mode = col_mode.item()
        col_min = float(df[col].min()) if not pd.isnull(df[col].min()) else None
        col_max = float(df[col].max()) if not pd.isnull(df[col].max()) else None
        col_std = float(df[col].std()) if not pd.isnull(df[col].std()) else None
        stats[col] = {
            "mean": col_mean,
            "median": col_median,
            "mode": col_mode,
            "min": col_min,
            "max": col_max,
            "std": col_std
        }
    return stats

@app.get("/api/correlations")
def get_correlations():
    """Return the correlation matrix for numeric columns."""
    df = get_data()
    corr = df.select_dtypes(include=[np.number]).corr()
    # Convert to dict for JSON
    return corr.round(2).to_dict() 