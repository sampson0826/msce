"""FastAPI server for Synthetic Data Decay Monitor.

Usage:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import json
import uuid
import sqlite3
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from synthetic_decay_monitor.data_lineage import (
    parse_lineage_from_jsonl, generate_synthetic_lineage,
)
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor, EmbeddingConstraintExtractor,
)
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.executor_classifier import (
    ExecutorClassifier, diagnose_executor_decay,
)
from synthetic_decay_monitor.stress_propagation import (
    run_cascade_analysis,
)
from synthetic_decay_monitor.report import (
    generate_json_report, generate_html_report, build_html_report,
)

app = FastAPI(
    title="Synthetic Data Decay Monitor",
    description="Constraint-Layer Health Diagnostic for AI Training Pipelines",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYSIS_CACHE: dict[str, dict] = {}

# ---- SQLite History DB ----
DB_PATH = os.environ.get("DECAY_MONITOR_DB", os.path.join(os.path.dirname(__file__), "history.db"))


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            n_samples INTEGER,
            n_generations INTEGER,
            runtime_seconds REAL,
            summary_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT,
            alert_type TEXT,
            capability TEXT,
            message TEXT,
            created_at TEXT,
            sent INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


_init_db()

# ---- Alerting Config ----
WEBHOOK_URL = os.environ.get("DECAY_MONITOR_WEBHOOK", "")
SLACK_WEBHOOK = os.environ.get("DECAY_MONITOR_SLACK_WEBHOOK", "")
ALERT_THRESHOLD_SEVERITY = float(os.environ.get("DECAY_MONITOR_ALERT_THRESHOLD", "0.6"))


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    n_samples: int
    n_generations: int
    n_capabilities: int
    trajectories: list
    diagnoses: list
    collapse_order: list
    cascade: Optional[dict]
    runtime_seconds: float
    report: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_jsonl(
    file: UploadFile = File(...),
    text_field: str = Query("text"),
    generation_field: str = Query("generation"),
    capability_field: str = Query("capability_tags"),
    format: str = Query("json", pattern="^(json|html)$"),
):
    """Upload JSONL data, run full decay analysis pipeline."""
    if not file.filename or not file.filename.endswith(('.jsonl', '.json')):
        raise HTTPException(400, "Please upload a .jsonl or .json file")

    t0 = time.time()
    content = await file.read()

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        lineage = parse_lineage_from_jsonl(
            tmp_path,
            text_field=text_field,
            generation_field=generation_field,
            capability_field=capability_field,
        )
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(400, f"Failed to parse JSONL: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Run full pipeline with Hybrid extractor
    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    # Executor diagnosis
    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(_serialize_diagnosis(diag))

    # Cascade analysis
    cascade = None
    if collapse_order:
        stability_map = {c["capability"]: c["current_S_n"] for c in collapse_order}
        cascade = run_cascade_analysis(collapse_order, stability_map)
        cascade = _serialize_cascade(cascade)

    # Generate report (proper json_report dict for both JSON and HTML paths)
    json_report = generate_json_report(
        lineage=lineage,
        trajectories=trajectories,
        diagnoses=diagnoses,
        cascade=cascade,
    )

    analysis_id = uuid.uuid4().hex[:12]
    runtime = time.time() - t0

    traj_serialized = []
    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            traj_serialized.append({
                "capability": t["capability"],
                "current_S_n": t.get("current_S_n", 0),
                "current_beta": t.get("current_beta", 0),
                "predicted_collapse_gen": t.get("predicted_collapse_gen", -1),
                "points": [
                    {"gen": p["generation"], "S_n": p["S_n"], "beta": p["beta"]}
                    for p in t["trajectory"]
                ],
            })

    result = {
        "analysis_id": analysis_id,
        "status": "completed",
        "n_samples": lineage.n_samples,
        "n_generations": lineage.n_generations,
        "n_capabilities": len(trajectories),
        "trajectories": traj_serialized,
        "diagnoses": diagnoses,
        "collapse_order": [
            {"capability": c["capability"], "current_S_n": c["current_S_n"],
             "predicted_collapse_gen": c.get("predicted_collapse_gen", -1)}
            for c in collapse_order
        ],
        "cascade": cascade,
        "runtime_seconds": round(runtime, 2),
    }

    if format == "html":
        result["report"] = build_html_report(json_report)

    ANALYSIS_CACHE[analysis_id] = result
    _auto_save(result)
    return result


@app.get("/report/{analysis_id}")
def get_report(analysis_id: str, format: str = Query("json", pattern="^(json|html)$")):
    """Retrieve a previously generated report."""
    result = ANALYSIS_CACHE.get(analysis_id)
    if not result:
        raise HTTPException(404, "Analysis not found")

    if format == "html":
        if "report" not in result:
            json_report = _rebuild_json_report(result)
            result["report"] = build_html_report(json_report)
        return HTMLResponse(result["report"])
    return result


@app.api_route("/analyze/demo", methods=["GET", "POST"])
async def analyze_demo(
    n_generations: int = Query(6, ge=2, le=20),
    format: str = Query("json", pattern="^(json|html)$"),
):
    """Run analysis with demo data (N generations of synthetic decay)."""
    t0 = time.time()
    demo_texts = [
        "The derivative of x^2 is 2x. To solve the equation, we apply the chain rule.",
        "Python functions are defined using the 'def' keyword. They can accept arguments and return values.",
        "The capital of France is Paris. The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
        "Therefore, based on the evidence presented, we can conclude that the hypothesis is supported.",
        "In the quiet village, the baker rose before dawn each day, kneading dough with hands that knew every curve.",
        "The water cycle involves evaporation, condensation, and precipitation. These processes are driven by solar energy.",
        "To optimize the algorithm, we can use dynamic programming to cache intermediate results and reduce complexity.",
        "The novel explores themes of identity and belonging through the eyes of a narrator who has lost both.",
        "Please provide a step-by-step solution showing all work and explaining the reasoning at each step.",
        "The study followed 10,000 participants over 20 years, tracking cardiovascular outcomes against dietary patterns.",
    ] * 3

    lineage = generate_synthetic_lineage(
        demo_texts, n_generations=n_generations, decay_pattern={"*": 0.25}
    )

    extractor = HybridConstraintExtractor(judge_fn=None)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    trajectories = engine.get_all_trajectories()
    collapse_order = engine.get_collapse_order()

    diagnoses = []
    for cap, snapshots in engine._snapshots.items():
        diag = diagnose_executor_decay(
            engine._trajectories.get(cap, []), snapshots, capability=cap
        )
        diagnoses.append(_serialize_diagnosis(diag))

    cascade = None
    if collapse_order:
        stability_map = {c["capability"]: c["current_S_n"] for c in collapse_order}
        cascade = run_cascade_analysis(collapse_order, stability_map)
        cascade = _serialize_cascade(cascade)

    analysis_id = uuid.uuid4().hex[:12]
    runtime = time.time() - t0

    traj_serialized = []
    for t in trajectories:
        if "trajectory" in t and t["trajectory"]:
            traj_serialized.append({
                "capability": t["capability"],
                "current_S_n": t.get("current_S_n", 0),
                "current_beta": t.get("current_beta", 0),
                "predicted_collapse_gen": t.get("predicted_collapse_gen", -1),
                "points": [
                    {"gen": p["generation"], "S_n": p["S_n"], "beta": p["beta"]}
                    for p in t["trajectory"]
                ],
            })

    result = {
        "analysis_id": analysis_id,
        "status": "completed",
        "n_samples": lineage.n_samples,
        "n_generations": lineage.n_generations,
        "n_capabilities": len(trajectories),
        "trajectories": traj_serialized,
        "diagnoses": diagnoses,
        "collapse_order": [
            {"capability": c["capability"], "current_S_n": c["current_S_n"],
             "predicted_collapse_gen": c.get("predicted_collapse_gen", -1)}
            for c in collapse_order
        ],
        "cascade": cascade,
        "runtime_seconds": round(runtime, 2),
    }

    if format == "html":
        json_report = generate_json_report(
            lineage=lineage, trajectories=trajectories,
            diagnoses=diagnoses, cascade=cascade,
        )
        result["report"] = build_html_report(json_report)

    ANALYSIS_CACHE[analysis_id] = result
    _auto_save(result)
    return result


@app.get("/capabilities")
def list_capabilities():
    """Return known capability executor priors."""
    from synthetic_decay_monitor.executor_classifier import CAPABILITY_EXECUTOR_PRIOR
    return {
        "executor_priors": CAPABILITY_EXECUTOR_PRIOR,
        "layers": {
            "E-I": "Axiom layer (α=0.40) — logic connector density, syntax CV",
            "E-II": "Scale layer (α=0.20) — bigram repetition, filler ratio, vocabulary diversity",
            "E-III": "Boundary layer (α=0.08) — proper noun case, number integrity",
        },
    }


def _serialize_diagnosis(diag: dict) -> dict:
    d = diag.get("diagnosis")
    return {
        "capability": diag.get("capability", ""),
        "degradation_type": d.degradation_type if hasattr(d, "degradation_type") else "unknown",
        "severity": d.severity if hasattr(d, "severity") else 0,
        "confidence": d.confidence if hasattr(d, "confidence") else 0,
        "intervention_type": d.intervention_type if hasattr(d, "intervention_type") else "",
        "intervention_urgency": diag.get("intervention_urgency", "monitor"),
        "current_S_n": diag.get("current_stability", 1.0),
        "current_beta": diag.get("current_beta", 0.0),
        "dark_zone_detected": diag.get("dark_zone_detected", False),
        "description": d.description if hasattr(d, "description") else "",
        "executor_composition": diag.get("executor_composition", {}),
        "recommended_data_sources": d.recommended_data_sources if hasattr(d, "recommended_data_sources") else [],
    }


def _serialize_cascade(cascade: dict) -> dict:
    return {
        "first_to_collapse": cascade.get("first_to_collapse", ""),
        "total_cascade_steps": cascade.get("total_cascade_steps", 0),
        "cascade_timeline": cascade.get("cascade_timeline", []),
        "weakest_edge": str(cascade.get("weakest_edge", {})),
        "stress_transfer_map": cascade.get("stress_transfer_map", {}),
    }


def _rebuild_json_report(result: dict) -> dict:
    """Rebuild a json_report dict from cached analysis result."""
    traj_list = _rebuild_trajectories(result.get("trajectories", []))
    return {
        "decay_analysis": {
            "trajectories": traj_list,
            "global_beta": _compute_global_beta(traj_list),
            "collapse_order": sorted(
                [t for t in traj_list if "error" not in t],
                key=lambda t: t.get("predicted_collapse_gen", 999),
            ),
        },
        "executor_diagnoses": result.get("diagnoses", []),
        "cascade_analysis": result.get("cascade", {}),
        "warnings": _generate_warnings_from_serialized(traj_list, result.get("diagnoses", [])),
        "data_summary": {
            "n_samples": result.get("n_samples", 0),
            "n_generations": result.get("n_generations", 0),
            "capabilities": [t["capability"] for t in traj_list],
        },
        "interventions": [],
    }


def _rebuild_trajectories(serialized: list) -> list:
    result = []
    for s in serialized:
        result.append({
            "capability": s["capability"],
            "trajectory": [
                {"generation": p["gen"], "S_n": p["S_n"], "beta": p["beta"]}
                for p in s.get("points", [])
            ],
            "current_S_n": s.get("current_S_n", 0),
            "current_beta": s.get("current_beta", 0),
            "predicted_collapse_gen": s.get("predicted_collapse_gen", -1),
        })
    return result


def _compute_global_beta(trajectories: list) -> float:
    betas = []
    for t in trajectories:
        if "trajectory" not in t:
            continue
        for g in t["trajectory"]:
            if "beta" in g and g["beta"] > 0:
                betas.append(g["beta"])
    return float(sum(betas) / len(betas)) if betas else 0.25


def _generate_warnings_from_serialized(trajectories: list, diagnoses: list) -> list:
    ws = []
    for t in trajectories:
        if "trajectory" not in t or not t["trajectory"]:
            continue
        last = t["trajectory"][-1]
        if last.get("status") == "collapsed":
            ws.append({"type": "capability_collapsed", "severity": "critical",
                       "capability": t["capability"],
                       "message": f"{t['capability']} has collapsed at generation {last['generation']}."})
        elif last.get("status") == "critical":
            ws.append({"type": "capability_critical", "severity": "warning",
                       "capability": t["capability"],
                       "message": f"{t['capability']} approaching collapse (S={last['S_n']:.3f} at gen {last['generation']})."})
    for d in diagnoses:
        if d.get("diagnosis", {}).get("degradation_type") == "E-I_loss":
            ws.append({"type": "EI_degradation", "severity": "critical",
                       "capability": d.get("capability", ""),
                       "message": f"{d.get('capability', '')}: E-I executor loss."})
    return ws


def _save_to_history(analysis_id: str, result: dict):
    """Persist analysis summary to SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO analyses (id, created_at, n_samples, n_generations, runtime_seconds, summary_json) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, datetime.now().isoformat(), result["n_samples"],
             result["n_generations"], result["runtime_seconds"],
             json.dumps(result.get("diagnoses", []), ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _check_alerts(analysis_id: str, result: dict):
    """Check and fire alerts for critical degradations."""
    alerts = []
    for diag in result.get("diagnoses", []):
        if diag.get("severity", 0) >= ALERT_THRESHOLD_SEVERITY:
            cap = diag["capability"]
            dtype = diag.get("degradation_type", "unknown")
            msg = (f"[Decay Monitor] {cap}: {dtype} "
                   f"severity={diag.get('severity', 0):.2f} "
                   f"S={diag.get('current_S_n', 1.0):.3f}")
            alerts.append({
                "analysis_id": analysis_id,
                "alert_type": dtype,
                "capability": cap,
                "message": msg,
            })

    if alerts:
        for alert in alerts:
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT INTO alerts (analysis_id, alert_type, capability, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (alert["analysis_id"], alert["alert_type"],
                     alert["capability"], alert["message"],
                     datetime.now().isoformat()),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        _fire_webhooks(alerts)


def _fire_webhooks(alerts: list):
    """Send alerts to configured webhooks."""
    payload = {
        "source": "decay-monitor",
        "timestamp": datetime.now().isoformat(),
        "alerts": alerts,
    }
    data = json.dumps(payload, ensure_ascii=False).encode()

    urls = [u for u in [WEBHOOK_URL, SLACK_WEBHOOK] if u]
    for url in urls:
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


@app.get("/history")
def get_history(limit: int = Query(20, ge=1, le=100)):
    """List recent analyses."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, created_at, n_samples, n_generations, runtime_seconds FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "created_at": r[1], "n_samples": r[2],
             "n_generations": r[3], "runtime_seconds": r[4]}
            for r in rows
        ]
    except Exception as e:
        return {"error": str(e)}


@app.get("/alerts")
def get_alerts(limit: int = Query(50, ge=1, le=500)):
    """List recent alerts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, analysis_id, alert_type, capability, message, created_at, sent FROM alerts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "analysis_id": r[1], "alert_type": r[2],
             "capability": r[3], "message": r[4], "created_at": r[5],
             "sent": bool(r[6])}
            for r in rows
        ]
    except Exception as e:
        return {"error": str(e)}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Simple HTML dashboard showing recent runs and alerts."""
    analyses = get_history(10)
    alerts = get_alerts(20)

    alert_rows = ""
    for a in alerts:
        alert_rows += f"<tr><td>{a['created_at'][:19]}</td><td>{a['capability']}</td><td>{a['alert_type']}</td><td>{a['message']}</td></tr>"

    analysis_rows = ""
    for a in analyses:
        analysis_rows += f"<tr><td><a href='/report/{a['id']}'>{a['id']}</a></td><td>{a['created_at'][:19]}</td><td>{a['n_samples']}</td><td>{a['n_generations']}</td><td>{a['runtime_seconds']}s</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Decay Monitor Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
.subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
.card {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
.card h2 {{ color: #38bdf8; margin-bottom: 1rem; font-size: 1.1rem; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #334155; font-size: 0.85rem; }}
th {{ color: #94a3b8; font-weight: 600; }}
a {{ color: #38bdf8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.alert-ei {{ color: #f87171; }}
.alert-eii {{ color: #fbbf24; }}
.alert-eiii {{ color: #a78bfa; }}
.actions {{ display: flex; gap: 1rem; margin-bottom: 2rem; }}
.btn {{ background: #1e40af; color: #e2e8f0; border: none; padding: 0.5rem 1.25rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }}
.btn:hover {{ background: #2563eb; }}
.status-bar {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
.stat {{ background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; text-align: center; }}
.stat-value {{ font-size: 1.5rem; font-weight: 700; color: #38bdf8; }}
.stat-label {{ font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }}
</style>
</head>
<body>
<h1>Synthetic Data Decay Monitor</h1>
<p class="subtitle">Constraint-Layer Health Diagnostic v0.2.0</p>

<div class="status-bar">
    <div class="stat"><div class="stat-value">{len(analyses)}</div><div class="stat-label">Recent Analyses</div></div>
    <div class="stat"><div class="stat-value">{len([a for a in alerts if a['alert_type'] == 'E-I_loss'])}</div><div class="stat-label">E-I Alerts</div></div>
    <div class="stat"><div class="stat-value">{len([a for a in alerts if a['alert_type'] == 'E-III_loss'])}</div><div class="stat-label">E-III Alerts</div></div>
</div>

<div class="actions" style="margin-top:1.5rem;">
    <button class="btn" onclick="fetch('/analyze/demo?n_generations=6&format=json').then(r=>r.json()).then(d=>{{alert('Done: '+d.n_samples+' samples, '+d.runtime_seconds+'s');location.reload()}})">Run Demo (6 gen)</button>
    <button class="btn" onclick="location.reload()">Refresh</button>
</div>

<div class="card">
    <h2>Recent Analyses</h2>
    <table>
        <tr><th>ID</th><th>Time</th><th>Samples</th><th>Gens</th><th>Runtime</th></tr>
        {analysis_rows or '<tr><td colspan="5">No analyses yet. Run a demo or upload data.</td></tr>'}
    </table>
</div>

<div class="card">
    <h2>Recent Alerts</h2>
    <table>
        <tr><th>Time</th><th>Capability</th><th>Type</th><th>Message</th></tr>
        {alert_rows or '<tr><td colspan="4">No alerts yet.</td></tr>'}
    </table>
</div>
</body>
</html>"""


def _auto_save(result: dict):
    """Auto-save analysis results and check alerts."""
    aid = result.get("analysis_id", "")
    if aid:
        _save_to_history(aid, result)
        _check_alerts(aid, result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
