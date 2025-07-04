import matplotlib
matplotlib.use('Agg')
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import backend.database as database
import time
from fastapi.responses import JSONResponse, Response
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

database.init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset1.csv')
CACHE_TTL = 300  # 5 minutes
cache = {
    "stats": {"data": None, "timestamp": 0},
    "predict": {"data": None, "timestamp": 0},
    "alerts": {"data": None, "timestamp": 0},
}

# Add a global list to store upload notifications
UPLOAD_NOTIFICATIONS = []

def is_cache_valid(key):
    return cache[key]["data"] is not None and (time.time() - cache[key]["timestamp"] < CACHE_TTL)

# Helper to get data from DB or CSV

def get_data():
    try:
        df = database.fetch_all_records()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Add notification for health data loaded (from DB)
            if not UPLOAD_NOTIFICATIONS or UPLOAD_NOTIFICATIONS[-1].get('type') != 'Load' or \
                (datetime.now() - datetime.strptime(UPLOAD_NOTIFICATIONS[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")).total_seconds() > 60:
                UPLOAD_NOTIFICATIONS.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Load",
                    "message": "Health data loaded successfully."
                })
            return df
    except Exception:
        pass
    df = pd.read_csv(DATA_PATH, encoding='latin1')
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
        "Battery_Level (%)": "battery_level",
        "Sleep (hrs)": "sleep",
        "Steps": "steps"
    })
    # Add notification for health data loaded (from CSV)
    if not UPLOAD_NOTIFICATIONS or UPLOAD_NOTIFICATIONS[-1].get('type') != 'Load' or \
        (datetime.now() - datetime.strptime(UPLOAD_NOTIFICATIONS[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")).total_seconds() > 60:
        UPLOAD_NOTIFICATIONS.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Load",
            "message": "Health data loaded successfully."
        })
    # Remap patient_id to sequential numbers starting from 1
    unique_ids = df['patient_id'].unique()
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    df['patient_id'] = df['patient_id'].map(id_map)
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
        "Target_Blood_Pressure", "Target_Heart_Rate", "Target_Health_Status", "Battery_Level (%)",
        "Sleep", "Steps"
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
        "heart_rate", "device_battery_level", "target_blood_pressure", "target_heart_rate", "target_health_status", "battery_level",
        "sleep", "steps"
    ]
    database.insert_records(df)
    # Invalidate cache
    for key in cache:
        cache[key]["data"] = None
        cache[key]["timestamp"] = 0
    # Add upload notification
    UPLOAD_NOTIFICATIONS.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "Upload",
        "message": "New data uploaded successfully."
    })
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
    # Only show non-health alerts (e.g., upload notifications)
    alerts_sorted = sorted(UPLOAD_NOTIFICATIONS, key=lambda x: str(x.get("timestamp", "")), reverse=True)
    result = {"alerts": alerts_sorted[:20]}
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

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.get("/api/eda/bar_chart")
def eda_bar_chart():
    df = get_data()
    fig, ax = plt.subplots(figsize=(6,4))
    df['sensor_type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Sensor Type Counts')
    ax.set_xlabel('Sensor Type')
    ax.set_ylabel('Count')
    img_base64 = plot_to_base64(fig)
    return {"image": img_base64}

@app.get("/api/eda/histogram")
def eda_histogram():
    df = get_data()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['heart_rate'].dropna(), bins=20, color='skyblue')
    ax.set_title('Heart Rate Distribution')
    ax.set_xlabel('Heart Rate')
    ax.set_ylabel('Frequency')
    img_base64 = plot_to_base64(fig)
    return {"image": img_base64}

@app.get("/api/eda/count_plot")
def eda_count_plot():
    df = get_data()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='target_health_status', data=df, ax=ax)
    ax.set_title('Target Health Status Count')
    img_base64 = plot_to_base64(fig)
    return {"image": img_base64}

@app.get("/api/eda/corr_heatmap")
def eda_corr_heatmap():
    df = get_data()
    fig, ax = plt.subplots(figsize=(16, 14))  # Larger figure
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        ax=ax,
        annot_kws={'size': 8},  # Smaller font for numbers
        fmt='.2f'
    )
    ax.set_title('Correlation Heatmap', fontsize=18)
    ax.tick_params(axis='x', labelrotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    fig.tight_layout()
    img_base64 = plot_to_base64(fig)
    return {"image": img_base64}

@app.get("/api/preprocessing_summary")
def preprocessing_summary():
    df = get_data()
    summary = {}
    # Missing values
    missing = df.isnull().sum().to_dict()
    summary['missing_values'] = missing
    # Label encoding for binary variables
    label_enc_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    label_encodings = {}
    for col in label_enc_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encodings[col] = list(le.classes_)
    summary['label_encoded'] = label_encodings
    # One-hot encoding for nominal variables
    nominal_cols = [col for col in df.select_dtypes(include='object').columns if col not in label_enc_cols and col != 'timestamp']
    onehot_cols = {}
    for col in nominal_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        transformed = ohe.fit_transform(df[[col]].astype(str))
        onehot_cols[col] = list(ohe.categories_[0])
    summary['onehot_encoded'] = onehot_cols
    # Feature scaling for numerical variables
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    summary['scaled_features'] = num_cols
    # Date fields
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        summary['date_features'] = ['year', 'month', 'day', 'hour', 'minute']
    # Target encoding
    if 'target_health_status' in df.columns:
        le = LabelEncoder()
        df['target_health_status_encoded'] = le.fit_transform(df['target_health_status'].astype(str))
        summary['target_encoding'] = list(le.classes_)
    return summary

@app.get("/api/feature_engineering")
def feature_engineering():
    df = get_data()
    features = {}
    # Example: extract date parts
    if 'timestamp' in df.columns:
        df['year'] = pd.to_datetime(df['timestamp']).dt.year
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['day'] = pd.to_datetime(df['timestamp']).dt.day
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['date_parts'] = ['year', 'month', 'day', 'hour']
    # Example: interaction term
    if 'heart_rate' in df.columns and 'temperature' in df.columns:
        df['hr_temp_interaction'] = df['heart_rate'] * df['temperature']
        features['interaction_terms'] = ['hr_temp_interaction']
    return features

@app.get("/api/predict_steps_per_patient")
def get_steps_prediction_per_patient():
    df = get_data()
    results = {}
    feature_cols = [
        "ordinal_time", "heart_rate", "temperature", "systolic_bp", "diastolic_bp", "device_battery_level", "battery_level"
    ]
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if group.shape[0] < 2:
            continue
        group["ordinal_time"] = group["timestamp"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") if isinstance(x, str) else x)
        group["ordinal_time"] = group["ordinal_time"].map(lambda x: x.toordinal())
        # Fill missing features with mean
        for col in feature_cols:
            if col not in group.columns:
                group[col] = group[col].mean()
        X = group[feature_cols].values
        y = group["steps"].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = group.iloc[-1]
        last_date = last_row["timestamp"]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
        preds = []
        for i in range(1, 8):
            future_date = last_date + timedelta(days=i)
            future_features = [
                future_date.toordinal(),
                last_row["heart_rate"],
                last_row["temperature"],
                last_row["systolic_bp"],
                last_row["diastolic_bp"],
                last_row["device_battery_level"],
                last_row["battery_level"]
            ]
            pred = model.predict([future_features])[0]
            preds.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_steps": round(pred, 2)
            })
        results[pid] = preds
    return {"predictions": results}

@app.get("/api/predict_sleep_per_patient")
def get_sleep_prediction_per_patient():
    df = get_data()
    if "sleep" not in df.columns:
        raise HTTPException(status_code=400, detail="'sleep' column not found in data.")
    results = {}
    feature_cols = [
        "ordinal_time", "heart_rate", "temperature", "systolic_bp", "diastolic_bp", "device_battery_level", "battery_level"
    ]
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if group.shape[0] < 2:
            continue
        group["ordinal_time"] = group["timestamp"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") if isinstance(x, str) else x)
        group["ordinal_time"] = group["ordinal_time"].map(lambda x: x.toordinal())
        # Fill missing features with mean
        for col in feature_cols:
            if col not in group.columns:
                group[col] = group[col].mean()
        X = group[feature_cols].values
        y = group["sleep"].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = group.iloc[-1]
        last_date = last_row["timestamp"]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
        preds = []
        for i in range(1, 8):
            future_date = last_date + timedelta(days=i)
            future_features = [
                future_date.toordinal(),
                last_row["heart_rate"],
                last_row["temperature"],
                last_row["systolic_bp"],
                last_row["diastolic_bp"],
                last_row["device_battery_level"],
                last_row["battery_level"]
            ]
            pred = model.predict([future_features])[0]
            preds.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_sleep": round(pred, 2)
            })
        results[pid] = preds
    return {"predictions": results}

@app.get("/api/predict_steps_7days")
def predict_steps_7days():
    df = get_data()
    step_cols = [f'Step_Day{i}' for i in range(1, 8)]
    for col in step_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")
    X_steps = df[step_cols].values
    from sklearn.ensemble import RandomForestRegressor
    model_steps = RandomForestRegressor(n_estimators=100, random_state=42)
    y_steps = df[step_cols].mean(axis=1)  # Dummy target
    model_steps.fit(X_steps, y_steps)
    pred_steps = model_steps.predict(X_steps)
    results = []
    for idx, row in df.iterrows():
        results.append({
            'patient_id': int(row['patient_id']) if 'patient_id' in row else idx+1,
            'predicted_step_day8': int(pred_steps[idx])
        })
    return {"predictions": results}

@app.get("/api/predict_sleep_7days")
def predict_sleep_7days():
    df = get_data()
    sleep_cols = [f'Sleep_Day{i}' for i in range(1, 8)]
    for col in sleep_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")
    X_sleep = df[sleep_cols].values
    from sklearn.ensemble import RandomForestRegressor
    model_sleep = RandomForestRegressor(n_estimators=100, random_state=42)
    y_sleep = df[sleep_cols].mean(axis=1)  # Dummy target
    model_sleep.fit(X_sleep, y_sleep)
    pred_sleep = model_sleep.predict(X_sleep)
    results = []
    for idx, row in df.iterrows():
        results.append({
            'patient_id': int(row['patient_id']) if 'patient_id' in row else idx+1,
            'predicted_sleep_day8': float(pred_sleep[idx])
        })
    return {"predictions": results} 