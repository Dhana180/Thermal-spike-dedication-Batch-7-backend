"""
ThermalSpike – Flask Backend
Single prediction endpoint: POST /api/predict
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, datetime, os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)

# ── Load model artefacts ──────────────────────────────────────────────────────
def load_models():
    with open(os.path.join(BASE, "models", "ohe.pkl"),      "rb") as f: ohe_cols      = pickle.load(f)
    with open(os.path.join(BASE, "models", "scaler.pkl"),   "rb") as f: all_feat_cols = pickle.load(f)
    with open(os.path.join(BASE, "models", "Xgboost.pkl"),  "rb") as f: model         = pickle.load(f)
    return ohe_cols, all_feat_cols, model

ohe_cols, all_feat_cols, model = load_models()

NUMERIC_COLS = [
    "user_session_duration_s","user_request_count","user_payload_size_mb",
    "user_cpu_cores_used","user_gpu_memory_used_gb","user_ram_used_gb",
    "user_disk_io_mbps","user_power_draw_w","user_cpu_contribution_pct",
    "user_gpu_contribution_pct","user_heat_contribution_pct",
    "inlet_temp_c","outlet_temp_c","hotspot_temp_c","cooling_capacity_pct",
    "airflow_rate_cfm","ambient_temp_c","humidity_pct","rolling_avg_temp_15m_c",
]
TIME_COLS = ["hour","day_of_week","month","is_weekend"]

def preprocess_input(inp):
    # Start with all numeric features
    row = {col: float(inp.get(col, 0.0)) for col in NUMERIC_COLS}

    # Initialise every column the model knows about to 0
    feat_list = list(all_feat_cols)          # ← convert once; may be ndarray / Index
    for col in feat_list:
        if col not in row:
            row[col] = 0

    # One-hot encode categorical fields
    for key in [
        f"ServerID_{inp.get('ServerID','').lower()}",
        f"UserID_{inp.get('UserID','').lower()}",
        f"DataCentreZone_{inp.get('DataCentreZone','').lower()}",
        f"WorkType_{inp.get('WorkType','').lower()}",
    ]:
        if key in row:
            row[key] = 1

    # Derive time features from timestamp
    ts_str = inp.get("Timestamp", str(datetime.datetime.now()))
    try:
        import pandas as pd
        ts = pd.to_datetime(ts_str)
    except Exception:
        ts = datetime.datetime.now()

    row["hour"]        = int(ts.hour)
    row["day_of_week"] = int(ts.weekday())
    row["month"]       = int(ts.month)
    row["is_weekend"]  = int(ts.weekday() >= 5)

    # ── FIX: use only feat_list as the column order ──────────────────────────
    # TIME_COLS are already inside all_feat_cols — do NOT append them again.
    # Appending caused duplicate columns → wrong feature-matrix width → the
    # "truth value of an array" error inside XGBoost.
    return np.array([[row.get(c, 0) for c in feat_list]], dtype=np.float64)


def build_causes_solutions(inp, pred, hotspot_temp, cooling_cap, power_draw,
                           airflow, rolling_avg, ambient_temp, work_type,
                           outlet_temp, inlet_temp, gpu_mem):
    causes, solutions = [], []
    if pred:          # pred is already a plain Python int — safe in bool context
        if hotspot_temp >= 80:
            causes.append(f"Critical hotspot temperature ({hotspot_temp}°C) — above safe threshold ~80°C")
            solutions.append("Immediately lower workload and activate emergency cooling mode")
        elif hotspot_temp >= 70:
            causes.append(f"Elevated hotspot temperature ({hotspot_temp}°C) — approaching critical threshold")
            solutions.append("Reduce workload intensity and increase cooling capacity")
        if cooling_cap < 50:
            causes.append(f"Insufficient cooling capacity ({cooling_cap}%) — cooling underperforming")
            solutions.append("Inspect CRAC units and chiller systems; check filter blockages")
        elif cooling_cap < 65:
            causes.append(f"Marginal cooling capacity ({cooling_cap}%) — insufficient for current heat load")
            solutions.append("Boost cooling fan speed and verify CRAC unit output")
        if power_draw >= 1200:
            causes.append(f"Extreme power draw ({power_draw}W) — generating excessive heat")
            solutions.append("Migrate high-power jobs to a cooler zone or distribute load")
        elif power_draw >= 900:
            causes.append(f"High power draw ({power_draw}W) — elevated thermal output")
            solutions.append("Throttle batch or ML-training jobs during peak thermal periods")
        if airflow < 1800:
            causes.append(f"Low airflow rate ({airflow} CFM) — inadequate heat dissipation")
            solutions.append("Clear rack obstructions; check blanking panels and CRAC airflow")
        if rolling_avg >= 75:
            causes.append(f"Sustained high rolling average ({rolling_avg}°C over 15 min)")
            solutions.append("Schedule a cooldown period; defer non-critical workloads")
        if ambient_temp >= 28:
            causes.append(f"High ambient temperature ({ambient_temp}°C) — reduces cooling delta")
            solutions.append("Check data centre HVAC system")
        if work_type in ["ML-training","batch","ETL","data-pipeline"] and power_draw >= 700:
            causes.append(f"Thermally intensive workload ({work_type}) with high resource utilisation")
            solutions.append(f"Schedule {work_type} jobs during off-peak hours or cooler zones")
        if gpu_mem >= 60:
            causes.append(f"High GPU memory utilisation ({gpu_mem}GB) — GPU at thermal limits")
            solutions.append("Use GPU memory optimisation: mixed precision, gradient checkpointing")
        if outlet_temp - inlet_temp >= 20:
            delta = round(outlet_temp - inlet_temp, 1)
            causes.append(f"Large inlet–outlet temperature delta ({delta}°C)")
            solutions.append("Review rack airflow layout; ensure hot/cold aisle containment")
        if not causes:
            causes.append("Combined sensor readings indicate elevated thermal risk")
            solutions.append("Perform a full thermal audit of the server and rack environment")
    return causes, solutions


# ── Single Prediction Endpoint ────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        X = preprocess_input(data)

        # ── FIX: force scalar Python types to avoid numpy bool ambiguity ──────
        raw_pred  = model.predict(X)
        pred      = int(np.array(raw_pred).flatten()[0])   # guaranteed Python int
        raw_proba = model.predict_proba(X)
        proba     = np.array(raw_proba).flatten().tolist()  # guaranteed Python list

        spike_pct  = round(proba[1] * 100, 2)
        normal_pct = round(proba[0] * 100, 2)

        hotspot_temp = float(data.get("hotspot_temp_c", 0))
        cooling_cap  = float(data.get("cooling_capacity_pct", 100))
        power_draw   = float(data.get("user_power_draw_w", 0))
        airflow      = float(data.get("airflow_rate_cfm", 3600))
        rolling_avg  = float(data.get("rolling_avg_temp_15m_c", 0))
        ambient_temp = float(data.get("ambient_temp_c", 25))
        work_type    = str(data.get("WorkType", ""))
        outlet_temp  = float(data.get("outlet_temp_c", 0))
        inlet_temp   = float(data.get("inlet_temp_c", 0))
        gpu_mem      = float(data.get("user_gpu_memory_used_gb", 0))

        causes, solutions = build_causes_solutions(
            data, pred, hotspot_temp, cooling_cap, power_draw,
            airflow, rolling_avg, ambient_temp, work_type,
            outlet_temp, inlet_temp, gpu_mem
        )

        return jsonify({
            "prediction":        pred,
            "is_spike":          bool(pred),
            "spike_probability": spike_pct,
            "normal_probability":normal_pct,
            "risk_band":         "High" if spike_pct >= 70 else ("Elevated" if spike_pct >= 40 else "Low"),
            "causes":            causes,
            "solutions":         solutions,
            "server_id":         data.get("ServerID",""),
            "zone":              data.get("DataCentreZone",""),
            "timestamp":         data.get("Timestamp", str(datetime.datetime.now())),
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())          # full trace in Render logs
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "XGBoost Thermal Spike Detector", "version": "1.0.0"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
