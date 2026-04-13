from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, datetime, os, warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ✅ CORS (Production Safe)
CORS(app)

# ✅ Base path
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load model artefacts safely ─────────────────────────
def load_models():
    try:
        with open(os.path.join(BASE, "models", "ohe.pkl"), "rb") as f:
            ohe_cols = pickle.load(f)

        with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
            all_feat_cols = pickle.load(f)

        with open(os.path.join(BASE, "models", "Xgboost.pkl"), "rb") as f:
            model = pickle.load(f)

        print("✅ Models loaded successfully")
        return ohe_cols, all_feat_cols, model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None, None

ohe_cols, all_feat_cols, model = load_models()

# ── Columns ─────────────────────────
NUMERIC_COLS = [
    "user_session_duration_s","user_request_count","user_payload_size_mb",
    "user_cpu_cores_used","user_gpu_memory_used_gb","user_ram_used_gb",
    "user_disk_io_mbps","user_power_draw_w","user_cpu_contribution_pct",
    "user_gpu_contribution_pct","user_heat_contribution_pct",
    "inlet_temp_c","outlet_temp_c","hotspot_temp_c","cooling_capacity_pct",
    "airflow_rate_cfm","ambient_temp_c","humidity_pct","rolling_avg_temp_15m_c",
]

TIME_COLS = ["hour","day_of_week","month","is_weekend"]

# ── Preprocess ─────────────────────────
def preprocess_input(inp):
    row = {col: float(inp.get(col, 0.0)) for col in NUMERIC_COLS}

    for col in (all_feat_cols or []):
        if col not in row:
            row[col] = 0

    for key in [
        f"ServerID_{inp.get('ServerID','').lower()}",
        f"UserID_{inp.get('UserID','').lower()}",
        f"DataCentreZone_{inp.get('DataCentreZone','').lower()}",
        f"WorkType_{inp.get('WorkType','').lower()}",
    ]:
        if key in row:
            row[key] = 1

    ts_str = inp.get("Timestamp", str(datetime.datetime.now()))

    try:
        import pandas as pd
        ts = pd.to_datetime(ts_str)
    except:
        ts = datetime.datetime.now()

    row["hour"] = ts.hour
    row["day_of_week"] = ts.weekday()
    row["month"] = ts.month
    row["is_weekend"] = int(ts.weekday() >= 5)

    ordered = list(all_feat_cols or []) + TIME_COLS

    return np.array([[row.get(c, 0) for c in ordered]])

# ── Cause/Solution Logic ─────────────────────────
def build_causes_solutions(inp, pred, hotspot_temp, cooling_cap, power_draw,
                          airflow, rolling_avg, ambient_temp, work_type,
                          outlet_temp, inlet_temp, gpu_mem):

    causes, solutions = [], []

    if pred:
        if hotspot_temp >= 80:
            causes.append(f"Critical hotspot temperature ({hotspot_temp}°C)")
            solutions.append("Reduce workload immediately")

        if cooling_cap < 50:
            causes.append(f"Low cooling capacity ({cooling_cap}%)")
            solutions.append("Check cooling systems")

        if power_draw >= 900:
            causes.append(f"High power draw ({power_draw}W)")
            solutions.append("Reduce load")

        if airflow < 1800:
            causes.append(f"Low airflow ({airflow} CFM)")
            solutions.append("Improve airflow")

    if not causes:
        causes.append("System operating normally")
        solutions.append("No action required")

    return causes, solutions

# ── ROOT ROUTE (FIXES NOT FOUND) ─────────────────────────
@app.route("/")
def home():
    return jsonify({
        "message": "Thermal Spike API Running 🚀",
        "status": "success"
    })

# ── HEALTH CHECK ─────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

# ── PREDICTION ─────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No input data"}), 400

        X = preprocess_input(data)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        spike_pct = round(proba[1] * 100, 2)
        normal_pct = round(proba[0] * 100, 2)

        causes, solutions = build_causes_solutions(
            data, pred,
            float(data.get("hotspot_temp_c", 0)),
            float(data.get("cooling_capacity_pct", 100)),
            float(data.get("user_power_draw_w", 0)),
            float(data.get("airflow_rate_cfm", 3600)),
            float(data.get("rolling_avg_temp_15m_c", 0)),
            float(data.get("ambient_temp_c", 25)),
            data.get("WorkType", ""),
            float(data.get("outlet_temp_c", 0)),
            float(data.get("inlet_temp_c", 0)),
            float(data.get("user_gpu_memory_used_gb", 0)),
        )

        return jsonify({
            "prediction": pred,
            "is_spike": bool(pred),
            "spike_probability": spike_pct,
            "normal_probability": normal_pct,
            "risk_band": "High" if spike_pct >= 70 else ("Elevated" if spike_pct >= 40 else "Low"),
            "causes": causes,
            "solutions": solutions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── RUN (FOR LOCAL ONLY) ─────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
