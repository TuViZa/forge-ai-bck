from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    pd = None


app = Flask(__name__)
# Limit uploads to 20 MB by default
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024))
CORS(app)

TOKENS: Dict[str, str] = {}


def project_root() -> str:
    # When app.py lives at repo root, this should be the repo root.
    # When app.py lives in backend/, this still returns the backend dir.
    return os.path.dirname(os.path.abspath(__file__))


def datasets_dir() -> str:
    # Priority:
    # 1) DATASETS_DIR env var
    # 2) <same-dir-as-app.py>/datasets
    # 3) <parent-of-app.py>/datasets
    env_dir = os.environ.get("DATASETS_DIR")
    if env_dir:
        return env_dir

    base = project_root()
    c1 = os.path.join(base, "datasets")
    if os.path.isdir(c1):
        return c1
    c2 = os.path.join(os.path.dirname(base), "datasets")
    if os.path.isdir(c2):
        return c2
    # Fallback to <same-dir-as-app.py>/datasets (may not exist yet; callers will create as needed)
    return c1


def _list_dataset_files() -> List[str]:
    ds = datasets_dir()
    if not os.path.isdir(ds):
        return []
    files: List[str] = []
    for name in os.listdir(ds):
        if name.lower().endswith(".csv") or name.lower().endswith(".xlsx"):
            files.append(os.path.join(ds, name))
    return files


def _read_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if pd is None:
        return None
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        # Read first sheet by default
        return pd.read_excel(path, engine="openpyxl")
    return None


def _compute_rows_for_files(files: List[str]) -> List[Dict]:
    if pd is None or not files:
        return []
    frames = []
    for p in files:
        try:
            df = _read_file(p)
            if df is not None:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return []
    big = pd.concat(frames, axis=0, ignore_index=True)
    norm, raw, canon = _normalize_columns(big)
    df = pd.DataFrame(norm)
    if df.get("Batch_ID") is None:
        df["Batch_ID"] = [f"BATCH-{str(i+1).zfill(3)}" for i in range(len(df))]
    else:
        df["Batch_ID"] = df["Batch_ID"].fillna("").astype(str)
        df.loc[df["Batch_ID"].eq("") | df["Batch_ID"].str.lower().eq("nan"), "Batch_ID"] = [
            f"BATCH-{str(i+1).zfill(3)}" for i in range(len(df))
        ]
    def num(col: str, default: float) -> "pd.Series":
        s = df[col]
        try:
            s = pd.to_numeric(s, errors="coerce")
        except Exception:
            s = pd.Series([None] * len(df))
        return s.fillna(default)
    df["Avg_Temperature"] = num("Avg_Temperature", 60.0)
    df["Max_Temperature"] = num("Max_Temperature", (df["Avg_Temperature"] + 10).clip(lower=0))
    df["Avg_Pressure"] = num("Avg_Pressure", 3.0)
    df["Avg_Power_Consumption"] = num("Avg_Power_Consumption", 20.0)
    df["Avg_Machine_Speed"] = num("Avg_Machine_Speed", 1100.0)
    df["Max_Compression_Force"] = num("Max_Compression_Force", 22.0)
    df["Batch_Duration"] = num("Batch_Duration", 6.0)
    df["Tablet_Hardness"] = num("Tablet_Hardness", 88.0)
    df["Dissolution_Rate"] = num("Dissolution_Rate", 92.0)
    df["Yield_Percentage"] = num("Yield_Percentage", 95.0)
    df["Quality_Score"] = num("Quality_Score", 90.0)
    df["Carbon_Footprint"] = (df["Avg_Power_Consumption"] * df["Batch_Duration"]).round(2)
    phases = ["Preparation", "Mixing", "Granulation", "Drying", "Compression", "Coating", "Packaging", "Quality Testing"]
    status_list: List[str] = []
    phase_list: List[str] = []
    progress_list: List[int] = []
    for _, row in df.iterrows():
        alert = row["Tablet_Hardness"] < 80 or row["Max_Temperature"] > 90
        status = "alert" if alert else "completed"
        if random.random() < 0.15:
            status = "in_progress"
        status_list.append(status)
        phase_list.append(random.choice(phases))
        progress_list.append(100 if status == "completed" else random.randint(5, 85))
    df["Status"] = status_list
    df["Phase"] = phase_list
    df["Progress"] = progress_list
    cols = [
        "Batch_ID",
        "Avg_Temperature",
        "Max_Temperature",
        "Avg_Pressure",
        "Avg_Power_Consumption",
        "Avg_Machine_Speed",
        "Max_Compression_Force",
        "Batch_Duration",
        "Tablet_Hardness",
        "Dissolution_Rate",
        "Yield_Percentage",
        "Quality_Score",
        "Carbon_Footprint",
        "Status",
        "Phase",
        "Progress",
    ]
    out_df = df[cols].copy().fillna(0)
    return out_df.to_dict(orient="records")


def _normalize_columns(df):
    # lower-case column names for flexible matching
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def pick(synonyms: List[str]) -> Optional[str]:
        for s in synonyms:
            if s in lower_map:
                return lower_map[s]
        # fuzzy contains match (e.g., 'temperature' in column name)
        for k in lower_map:
            if any(s in k for s in synonyms):
                return lower_map[k]
        return None

    # establish a best-effort canonical column set
    mapping: Dict[str, List[str]] = {
        "Batch_ID": ["batch_id", "batch id", "batchid", "batch"],
        "Avg_Temperature": ["avg_temperature", "average temperature", "temperature avg", "avg temp", "temperature"],
        "Max_Temperature": ["max_temperature", "maximum temperature", "max temp"],
        "Avg_Pressure": ["avg_pressure", "average pressure", "pressure avg", "pressure"],
        "Avg_Power_Consumption": ["avg_power_consumption", "average power", "power avg", "power"],
        "Avg_Machine_Speed": ["avg_machine_speed", "machine speed", "speed"],
        "Max_Compression_Force": ["max_compression_force", "compression force", "max force", "force"],
        "Batch_Duration": ["batch_duration", "duration", "time", "total time"],
        "Tablet_Hardness": ["tablet_hardness", "hardness"],
        "Dissolution_Rate": ["dissolution_rate", "dissolution"],
        "Yield_Percentage": ["yield_percentage", "yield %", "yield"],
        "Quality_Score": ["quality_score", "quality", "score"],
    }

    canon: Dict[str, Optional[str]] = {}
    for target, syns in mapping.items():
        canon[target] = pick(syns)

    # Build result with safe defaults
    out = {}
    for col, src in canon.items():
        if src and src in df.columns:
            out[col] = df[src]
        else:
            out[col] = None

    return out, df, canon


def _compute_rows() -> List[Dict]:
    files = _list_dataset_files()
    return _compute_rows_for_files(files)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/batches")
def api_batches():
    try:
        token = request.args.get("token")
        if token and token in TOKENS:
            data = _compute_rows_for_files([TOKENS[token]])
        else:
            data = _compute_rows()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/alerts")
def api_alerts():
    try:
        token = request.args.get("token")
        if token and token in TOKENS:
            rows = _compute_rows_for_files([TOKENS[token]])
        else:
            rows = _compute_rows()
        alerts = []
        for r in rows:
            reasons = []
            if r.get("Max_Temperature", 0) > 90:
                reasons.append(f"High Temp: {r.get('Max_Temperature')}°C")
            if r.get("Tablet_Hardness", 0) < 80:
                reasons.append(f"Low Hardness: {r.get('Tablet_Hardness')}")
            if r.get("Status") == "alert" and not reasons:
                reasons.append("Status flagged as alert")
            if reasons:
                alerts.append(
                    {
                        "batchId": r.get("Batch_ID"),
                        "reason": reasons[0],
                    }
                )
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/predictions")
def api_predictions():
    try:
        token = request.args.get("token")
        if token and token in TOKENS:
            rows = _compute_rows_for_files([TOKENS[token]])
        else:
            rows = _compute_rows()
        if not rows:
            return jsonify(
                {
                    "batchId": None,
                    "predictedQuality": 0.0,
                    "predictedEnergy": 0.0,
                    "predictedCO2": 0.0,
                }
            )
        # choose an in-progress batch if available, else first
        candidates = [r for r in rows if r.get("Status") == "in_progress"]
        target = candidates[0] if candidates else rows[0]
        power = float(target.get("Avg_Power_Consumption", 20.0) or 20.0)
        hardness = float(target.get("Tablet_Hardness", 90.0) or 90.0)
        yield_pct = float(target.get("Yield_Percentage", 95.0) or 95.0)
        pressure = float(target.get("Avg_Pressure", 3.0) or 3.0)

        # heuristic prediction
        q = 88.0 + (hardness - 90.0) * 0.25 + (yield_pct - 95.0) * 0.15 - (power - 20.0) * 0.2
        q = max(70.0, min(99.0, round(q, 1)))
        e = power + (pressure - 3.0) * 0.6
        e = max(10.0, min(35.0, round(e, 1)))
        co2 = round(e * 6.2, 1)

        return jsonify(
            {
                "batchId": target.get("Batch_ID"),
                "predictedQuality": q,
                "predictedEnergy": e,
                "predictedCO2": co2,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/upload")
def api_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file field required"}), 400
        f = request.files["file"]
        if f.filename is None or f.filename == "":
            return jsonify({"error": "empty filename"}), 400
        name = os.path.basename(f.filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".csv", ".xlsx"):
            return jsonify({"error": "only .csv and .xlsx allowed"}), 400
        up_dir = os.path.join(datasets_dir(), "uploads")
        os.makedirs(up_dir, exist_ok=True)
        save_path = os.path.join(up_dir, name)
        f.save(save_path)
        token = str(abs(hash(save_path)))
        TOKENS[token] = save_path
        return jsonify({"ok": True, "token": token, "filename": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
