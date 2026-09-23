import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any

from app.model_engine import engine

app = FastAPI(
    title="Wildfire & Flood Prediction System",
    description="Team 56 - Deep Learning Project: Feature-Attention Neural Network & LSTM Forecasting Dashboard",
    version="2.0.0"
)

class WildfireInput(BaseModel):
    Temperature: float = 34.0
    RH: float = 45.0
    Ws: float = 18.0
    Rain: float = 0.0
    FFMC: float = 85.0
    DMC: float = 25.0
    DC: float = 110.0
    ISI: float = 9.5
    BUI: float = 30.0
    FWI: float = 14.0

class FloodInput(BaseModel):
    MonsoonIntensity: float = 7.5
    TopographyDrainage: float = 4.0
    RiverManagement: float = 3.5
    Deforestation: float = 6.0
    Urbanization: float = 7.0
    ClimateChange: float = 6.5
    DamsQuality: float = 4.0
    Siltation: float = 6.0
    DrainageSystems: float = 3.5
    Rainfall: float = 180.0

class SequenceInput(BaseModel):
    dry_spell_days: int = 10
    init_temp: float = 36.0

@app.post("/api/predict/wildfire")
def predict_wildfire(data: WildfireInput):
    prob, att = engine.predict_wildfire(data.dict())
    risk_pct = round(prob * 100, 1)
    if risk_pct < 25:
        category = "Low Risk"
        color = "#00E676"
        level = 1
    elif risk_pct < 55:
        category = "Moderate Risk"
        color = "#FFD600"
        level = 2
    elif risk_pct < 80:
        category = "High Risk"
        color = "#FF6D00"
        level = 3
    else:
        category = "Severe Risk"
        color = "#FF1744"
        level = 4
    return {
        "risk_probability": prob,
        "risk_percentage": risk_pct,
        "category": category,
        "color": color,
        "level": level,
        "attention_weights": att
    }

@app.post("/api/predict/flood")
def predict_flood(data: FloodInput):
    prob, att = engine.predict_flood(data.dict())
    risk_pct = round(prob * 100, 1)
    if risk_pct < 25:
        category = "Low Risk"
        color = "#00E676"
        level = 1
    elif risk_pct < 55:
        category = "Moderate Risk"
        color = "#00B0FF"
        level = 2
    elif risk_pct < 80:
        category = "High Risk"
        color = "#2979FF"
        level = 3
    else:
        category = "Severe Flood Risk"
        color = "#AA00FF"
        level = 4
    return {
        "risk_probability": prob,
        "risk_percentage": risk_pct,
        "category": category,
        "color": color,
        "level": level,
        "attention_weights": att
    }

@app.post("/api/predict/sequence")
def predict_sequence(data: SequenceInput):
    res = engine.predict_sequence(dry_spell_days=data.dry_spell_days, init_temp=data.init_temp)
    return res

@app.get("/api/metrics")
def get_metrics():
    return {
        "wildfire": [
            {"model": "Logistic Regression", "accuracy": 0.947, "precision": 0.972, "recall": 0.921, "f1": 0.946, "auc": 0.985},
            {"model": "Random Forest", "accuracy": 0.887, "precision": 0.903, "recall": 0.868, "f1": 0.885, "auc": 0.964},
            {"model": "Baseline MLP (NN)", "accuracy": 0.930, "precision": 0.951, "recall": 0.907, "f1": 0.929, "auc": 0.985},
            {"model": "Feature-Attention NN (Ours)", "accuracy": 0.930, "precision": 0.958, "recall": 0.901, "f1": 0.928, "auc": 0.984}
        ],
        "flood": [
            {"model": "Logistic Regression", "accuracy": 0.940, "precision": 0.949, "recall": 0.933, "f1": 0.941, "auc": 0.986},
            {"model": "Random Forest", "accuracy": 0.880, "precision": 0.878, "recall": 0.888, "f1": 0.883, "auc": 0.960},
            {"model": "Baseline MLP (NN)", "accuracy": 0.923, "precision": 0.947, "recall": 0.899, "f1": 0.923, "auc": 0.980},
            {"model": "Feature-Attention NN (Ours)", "accuracy": 0.931, "precision": 0.938, "recall": 0.927, "f1": 0.933, "auc": 0.984}
        ],
        "sequence": [
            {"model": "LSTM (Full 12-day sequence history)", "input": "Sequential (K=12)", "accuracy": 0.719, "f1": 0.762, "auc": 0.771},
            {"model": "MLP Baseline", "input": "Single-day only", "accuracy": 0.629, "f1": 0.727, "auc": 0.592},
            {"model": "Logistic Regression", "input": "Single-day only", "accuracy": 0.638, "f1": 0.734, "auc": 0.568}
        ]
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DisasterAI — Wildfire & Flood Intelligence Platform</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#03060f;
  --surface:rgba(8,16,32,0.85);
  --surface2:rgba(12,22,44,0.9);
  --border:rgba(255,255,255,0.07);
  --border-bright:rgba(255,255,255,0.14);
  --fire:#FF4500;
  --fire2:#FF8C00;
  --flood:#0077FF;
  --flood2:#00C8FF;
  --green:#00E676;
  --yellow:#FFD600;
  --text:#E8EDF5;
  --muted:#607B96;
  --mono:'JetBrains Mono',monospace;
}
html{scroll-behavior:smooth}
body{
  background:var(--bg);color:var(--text);
  font-family:'Inter',sans-serif;min-height:100vh;
  overflow-x:hidden;
}

/* ── PARTICLE CANVAS ── */
#particle-canvas{
  position:fixed;top:0;left:0;width:100%;height:100%;
  pointer-events:none;z-index:0;opacity:.55;
}

/* ── LAYOUT ── */
.app-shell{position:relative;z-index:1;display:flex;flex-direction:column;min-height:100vh;}

/* ── HEADER ── */
header{
  display:flex;justify-content:space-between;align-items:center;
  padding:1rem 2.5rem;
  background:rgba(3,6,15,0.8);
  backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{
  width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,var(--fire),var(--flood));
  display:flex;align-items:center;justify-content:center;font-size:22px;
  box-shadow:0 0 24px rgba(255,69,0,.35),0 0 48px rgba(0,119,255,.2);
  animation:pulse-logo 3s ease-in-out infinite;
}
@keyframes pulse-logo{0%,100%{box-shadow:0 0 24px rgba(255,69,0,.35),0 0 48px rgba(0,119,255,.2)}50%{box-shadow:0 0 36px rgba(255,69,0,.6),0 0 64px rgba(0,119,255,.35)}}
.logo-text h1{font-size:1.15rem;font-weight:800;letter-spacing:-.5px}
.logo-text span{font-size:.72rem;color:var(--muted);font-family:var(--mono)}
.status-pill{
  display:flex;align-items:center;gap:7px;
  padding:6px 14px;border-radius:20px;
  background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.2);
  font-size:.75rem;font-weight:600;color:var(--green);
}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 1.4s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* ── NAV ── */
nav{display:flex;gap:4px;background:rgba(255,255,255,.03);padding:4px;border-radius:12px;border:1px solid var(--border)}
.nav-btn{
  background:transparent;border:none;color:var(--muted);
  padding:8px 18px;border-radius:9px;font-weight:600;font-size:.82rem;
  cursor:pointer;transition:all .2s;letter-spacing:.2px;
}
.nav-btn:hover{color:var(--text);background:rgba(255,255,255,.05)}
.nav-btn.active{color:#fff;background:linear-gradient(135deg,rgba(255,69,0,.25),rgba(0,119,255,.2));border:1px solid var(--border-bright)}

/* ── ALERT BANNER ── */
#alert-banner{
  display:none;margin:1.5rem 2.5rem 0;
  border-radius:14px;padding:1rem 1.5rem;
  border-left:4px solid;
  backdrop-filter:blur(12px);
  animation:slide-in .4s ease;
}
@keyframes slide-in{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
.alert-inner{display:flex;align-items:center;gap:12px}
.alert-icon{font-size:1.6rem}
.alert-text strong{display:block;font-size:.95rem;font-weight:700;margin-bottom:2px}
.alert-text span{font-size:.8rem;color:var(--muted)}

/* ── MAIN ── */
main{flex:1;padding:2rem 2.5rem;max-width:1440px;margin:0 auto;width:100%}
.tab-content{display:none;animation:fade-in .4s ease}
.tab-content.active{display:block}
@keyframes fade-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ── KPI STRIP ── */
.kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.75rem}
.kpi-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:16px;padding:1.25rem 1.5rem;
  backdrop-filter:blur(16px);position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s;
}
.kpi-card:hover{transform:translateY(-2px);border-color:var(--border-bright)}
.kpi-card::before{content:'';position:absolute;top:0;right:0;width:80px;height:80px;border-radius:50%;filter:blur(30px);opacity:.25}
.kpi-card.fire::before{background:var(--fire)}
.kpi-card.flood::before{background:var(--flood)}
.kpi-card.green::before{background:var(--green)}
.kpi-card.yellow::before{background:var(--yellow)}
.kpi-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:.5rem}
.kpi-value{font-size:2rem;font-weight:800;font-family:var(--mono);line-height:1}
.kpi-sub{font-size:.75rem;color:var(--muted);margin-top:.35rem}

/* ── HERO BAND ── */
.hero-band{
  background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:1.5rem 2rem;margin-bottom:1.75rem;
  backdrop-filter:blur(16px);
  display:flex;align-items:center;gap:1.5rem;
  position:relative;overflow:hidden;
}
.hero-band::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,69,0,.06) 0%,transparent 60%,rgba(0,119,255,.06) 100%);
  pointer-events:none;
}
.hero-badge{
  padding:5px 12px;border-radius:20px;font-size:.72rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.6px;
  background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);
}
.hero-band h2{font-size:1.55rem;font-weight:800;margin:.4rem 0 .4rem}
.hero-band p{color:var(--muted);font-size:.88rem;line-height:1.6;max-width:700px}

/* ── GRID ── */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.25rem}

/* ── CARD ── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:1.75rem;
  backdrop-filter:blur(16px);
  transition:border-color .2s;
}
.card:hover{border-color:var(--border-bright)}
.card-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem}
.card-title{font-size:.95rem;font-weight:700;letter-spacing:-.2px}
.card-badge{
  font-size:.68rem;font-weight:700;padding:3px 9px;border-radius:6px;
  background:rgba(255,255,255,.06);color:var(--muted);border:1px solid var(--border);
}

/* ── CONTROL SLIDERS ── */
.ctrl-group{margin-bottom:1rem}
.ctrl-label{display:flex;justify-content:space-between;font-size:.78rem;font-weight:500;margin-bottom:5px;color:var(--muted)}
.ctrl-val{font-family:var(--mono);font-size:.78rem;color:var(--text);background:rgba(255,255,255,.05);padding:2px 8px;border-radius:5px}
input[type=range]{
  width:100%;height:4px;border-radius:2px;
  background:rgba(255,255,255,.08);outline:none;
  -webkit-appearance:none;cursor:pointer;
  accent-color:var(--fire);
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;border-radius:50%;
  background:#fff;box-shadow:0 0 8px rgba(255,255,255,.5);cursor:pointer;
}

/* ── SVG GAUGE ── */
.gauge-wrap{display:flex;flex-direction:column;align-items:center;padding:1rem 0 .5rem}
.gauge-svg-wrap{position:relative;width:200px;height:120px}
.gauge-svg-wrap svg{overflow:visible}
.gauge-pct{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-size:2.8rem;font-weight:900;font-family:var(--mono);line-height:1;
  text-align:center;
}
.gauge-cat{
  margin-top:.75rem;padding:6px 20px;border-radius:20px;
  font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
  transition:all .3s;
}

/* ── CHART BOX ── */
.chart-box{position:relative;height:240px;width:100%}
.chart-box-tall{position:relative;height:300px;width:100%}

/* ── RISK METER BARS ── */
.risk-meter{margin-top:1rem}
.risk-meter-bar-wrap{display:flex;align-items:center;gap:10px;margin-bottom:.6rem}
.risk-meter-label{font-size:.75rem;color:var(--muted);width:130px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.risk-meter-track{flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,.07);overflow:hidden}
.risk-meter-fill{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.risk-meter-pct{font-size:.72rem;font-family:var(--mono);color:var(--text);width:38px;text-align:right}

/* ── METRICS TABLE ── */
.tbl-wrap{overflow-x:auto;margin-top:1rem}
table{width:100%;border-collapse:collapse}
th,td{padding:11px 16px;text-align:left;font-size:.82rem;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;background:rgba(255,255,255,.02)}
tr.ours{background:rgba(255,69,0,.06)}
tr.ours td:first-child{border-left:3px solid var(--fire)}
.badge-ours{
  background:linear-gradient(90deg,rgba(255,69,0,.2),rgba(255,140,0,.2));
  color:var(--fire2);border:1px solid rgba(255,69,0,.3);
  padding:2px 8px;border-radius:5px;font-size:.7rem;font-weight:700;margin-left:6px;
}
.metric-good{color:var(--green);font-weight:700}
.metric-best{color:var(--green);font-weight:800;font-size:.88rem}

/* ── LSTM HEATMAP ── */
.heatmap-row{display:flex;gap:4px;margin:1rem 0}
.heat-cell{
  flex:1;border-radius:8px;height:48px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-size:.65rem;font-weight:600;color:#fff;
  transition:all .4s;cursor:default;
}
.heat-cell:hover{transform:scaleY(1.15);z-index:2}
.heat-cell .day-lbl{font-size:.6rem;opacity:.7;margin-top:2px}

/* ── DAY CARDS ── */
.day-cards{display:grid;grid-template-columns:repeat(6,1fr);gap:.6rem;margin-top:1rem}
.day-card{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:12px;padding:.75rem .5rem;text-align:center;
  transition:all .3s;
}
.day-card:hover{background:rgba(255,255,255,.06);transform:translateY(-2px)}
.day-card .dc-label{font-size:.65rem;color:var(--muted);margin-bottom:.35rem}
.day-card .dc-pct{font-size:1.1rem;font-weight:800;font-family:var(--mono)}
.day-card .dc-bar{height:3px;border-radius:2px;margin-top:.4rem;transition:width .5s}

/* ── SECTION TITLE ── */
.sec-title{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:.75rem;display:flex;align-items:center;gap:8px}
.sec-title::after{content:'';flex:1;height:1px;background:var(--border)}

/* ── FOOTER ── */
footer{
  text-align:center;padding:1.5rem;
  border-top:1px solid var(--border);
  color:var(--muted);font-size:.75rem;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:3px}

/* ── RESPONSIVE ── */
@media(max-width:900px){
  .grid-2,.kpi-strip{grid-template-columns:1fr}
  .day-cards{grid-template-columns:repeat(4,1fr)}
  header{flex-wrap:wrap;gap:.75rem}
  nav{order:3;width:100%;justify-content:center}
}
</style>
</head>
<body>

<canvas id="particle-canvas"></canvas>

<div class="app-shell">

<header>
  <div class="logo">
    <div class="logo-icon">🔥</div>
    <div class="logo-text">
      <h1>DisasterAI</h1>
      <span>Wildfire &amp; Flood Intelligence · Team 56</span>
    </div>
  </div>
  <nav>
    <button class="nav-btn active" onclick="switchTab('wildfire',event)">🔥 Wildfire</button>
    <button class="nav-btn" onclick="switchTab('flood',event)">🌊 Flood</button>
    <button class="nav-btn" onclick="switchTab('lstm',event)">📈 LSTM Forecast</button>
    <button class="nav-btn" onclick="switchTab('metrics',event)">📊 Benchmarks</button>
  </nav>
  <div class="status-pill"><div class="status-dot"></div>Models Live</div>
</header>

<!-- ALERT BANNER -->
<div id="alert-banner">
  <div class="alert-inner">
    <div class="alert-icon" id="alert-icon">⚠️</div>
    <div class="alert-text">
      <strong id="alert-title">High Risk Detected</strong>
      <span id="alert-desc">Adjust parameters to explore risk factors.</span>
    </div>
  </div>
</div>

<main>

<!-- ═══════════════ WILDFIRE TAB ═══════════════ -->
<div id="tab-wildfire" class="tab-content active">

  <div class="kpi-strip" id="w-kpi-strip">
    <div class="kpi-card fire">
      <div class="kpi-label">Fire Risk</div>
      <div class="kpi-value" id="w-kpi-risk" style="color:var(--fire)">--%</div>
      <div class="kpi-sub" id="w-kpi-cat">Calculating...</div>
    </div>
    <div class="kpi-card yellow">
      <div class="kpi-label">Top Driver</div>
      <div class="kpi-value" id="w-kpi-driver" style="color:var(--yellow);font-size:1.1rem;padding-top:.3rem">—</div>
      <div class="kpi-sub">Highest attention weight</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">Model Accuracy</div>
      <div class="kpi-value" style="color:var(--green)">93.0%</div>
      <div class="kpi-sub">FANN · Test set</div>
    </div>
    <div class="kpi-card flood">
      <div class="kpi-label">AUC-ROC</div>
      <div class="kpi-value" style="color:var(--flood2)">0.984</div>
      <div class="kpi-sub">Feature-Attention NN</div>
    </div>
  </div>

  <div class="hero-band">
    <div>
      <div class="hero-badge">Feature-Attention Neural Network</div>
      <h2>Wildfire Hazard Risk Predictor</h2>
      <p>Tune 10 environmental parameters in real-time. The FANN produces a calibrated fire probability with learnable attention weights showing which factors are driving the prediction.</p>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-hdr"><div class="card-title">Environmental Parameters</div><div class="card-badge">10 features</div></div>
      <div id="wildfire-controls"></div>
    </div>

    <div class="card">
      <div class="card-hdr"><div class="card-title">FANN Risk Assessment</div></div>
      <div class="gauge-wrap">
        <div class="gauge-svg-wrap">
          <svg width="200" height="120" viewBox="0 0 200 120">
            <defs>
              <linearGradient id="wGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#00E676"/>
                <stop offset="50%" stop-color="#FFD600"/>
                <stop offset="100%" stop-color="#FF1744"/>
              </linearGradient>
            </defs>
            <!-- track -->
            <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="14" stroke-linecap="round"/>
            <!-- fill -->
            <path id="w-gauge-arc" d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="url(#wGrad)" stroke-width="14" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="251.2" style="transition:stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)"/>
            <!-- needle -->
            <line id="w-needle" x1="100" y1="110" x2="100" y2="38" stroke="#fff" stroke-width="2.5" stroke-linecap="round" style="transform-origin:100px 110px;transition:transform .8s cubic-bezier(.4,0,.2,1)"/>
            <circle cx="100" cy="110" r="5" fill="#fff"/>
          </svg>
          <div class="gauge-pct" id="w-gauge-pct" style="color:var(--fire)">0%</div>
        </div>
        <div class="gauge-cat" id="w-gauge-cat">Calculating...</div>
      </div>

      <div class="sec-title">Attention Weight Distribution</div>
      <div class="risk-meter" id="w-att-bars"></div>
    </div>
  </div>
</div>

<!-- ═══════════════ FLOOD TAB ═══════════════ -->
<div id="tab-flood" class="tab-content">

  <div class="kpi-strip">
    <div class="kpi-card flood">
      <div class="kpi-label">Flood Risk</div>
      <div class="kpi-value" id="f-kpi-risk" style="color:var(--flood2)">--%</div>
      <div class="kpi-sub" id="f-kpi-cat">Calculating...</div>
    </div>
    <div class="kpi-card yellow">
      <div class="kpi-label">Top Driver</div>
      <div class="kpi-value" id="f-kpi-driver" style="color:var(--yellow);font-size:1.1rem;padding-top:.3rem">—</div>
      <div class="kpi-sub">Highest attention weight</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">Model Accuracy</div>
      <div class="kpi-value" style="color:var(--green)">93.1%</div>
      <div class="kpi-sub">FANN · Test set</div>
    </div>
    <div class="kpi-card fire">
      <div class="kpi-label">AUC-ROC</div>
      <div class="kpi-value" style="color:var(--fire2)">0.984</div>
      <div class="kpi-sub">Feature-Attention NN</div>
    </div>
  </div>

  <div class="hero-band">
    <div>
      <div class="hero-badge" style="background:rgba(0,119,255,.1);color:var(--flood2);border-color:rgba(0,119,255,.25)">Hydrological Attention Model</div>
      <h2>Flood Hazard Risk Predictor</h2>
      <p>Adjust hydrological and land-use parameters. The model reveals which topographic and climatic factors contribute most to flood probability through learnable attention.</p>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-hdr"><div class="card-title">Hydrological Parameters</div><div class="card-badge">10 features</div></div>
      <div id="flood-controls"></div>
    </div>

    <div class="card">
      <div class="card-hdr"><div class="card-title">FANN Flood Risk Assessment</div></div>
      <div class="gauge-wrap">
        <div class="gauge-svg-wrap">
          <svg width="200" height="120" viewBox="0 0 200 120">
            <defs>
              <linearGradient id="fGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#00E676"/>
                <stop offset="50%" stop-color="#00B0FF"/>
                <stop offset="100%" stop-color="#AA00FF"/>
              </linearGradient>
            </defs>
            <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="14" stroke-linecap="round"/>
            <path id="f-gauge-arc" d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="url(#fGrad)" stroke-width="14" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="251.2" style="transition:stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)"/>
            <line id="f-needle" x1="100" y1="110" x2="100" y2="38" stroke="#fff" stroke-width="2.5" stroke-linecap="round" style="transform-origin:100px 110px;transition:transform .8s cubic-bezier(.4,0,.2,1)"/>
            <circle cx="100" cy="110" r="5" fill="#fff"/>
          </svg>
          <div class="gauge-pct" id="f-gauge-pct" style="color:var(--flood2)">0%</div>
        </div>
        <div class="gauge-cat" id="f-gauge-cat">Calculating...</div>
      </div>

      <div class="sec-title">Attention Weight Distribution</div>
      <div class="risk-meter" id="f-att-bars"></div>
    </div>
  </div>
</div>

<!-- ═══════════════ LSTM TAB ═══════════════ -->
<div id="tab-lstm" class="tab-content">

  <div class="kpi-strip">
    <div class="kpi-card fire">
      <div class="kpi-label">Peak Risk (Day 12)</div>
      <div class="kpi-value" id="lstm-kpi-peak" style="color:var(--fire)">--%</div>
      <div class="kpi-sub">LSTM final prediction</div>
    </div>
    <div class="kpi-card yellow">
      <div class="kpi-label">Risk Trend</div>
      <div class="kpi-value" id="lstm-kpi-trend" style="color:var(--yellow);font-size:1.3rem;padding-top:.2rem">—</div>
      <div class="kpi-sub">Day 1 → Day 12</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">LSTM AUC-ROC</div>
      <div class="kpi-value" style="color:var(--green)">0.771</div>
      <div class="kpi-sub">vs 0.568 baseline</div>
    </div>
    <div class="kpi-card flood">
      <div class="kpi-label">Lookback Window</div>
      <div class="kpi-value" style="color:var(--flood2)">K = 12</div>
      <div class="kpi-sub">Sequential context days</div>
    </div>
  </div>

  <div class="hero-band">
    <div>
      <div class="hero-badge">12-Day Sequential LSTM</div>
      <h2>Multi-Day Dry-Spell Risk Buildup Forecast</h2>
      <p>The LSTM reads 12 consecutive days of weather context, capturing temporal dynamics that single-day models miss entirely. Watch risk accumulate as dry-spell duration and temperature increase.</p>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-hdr"><div class="card-title">Scenario Controls</div></div>
      <div class="ctrl-group">
        <div class="ctrl-label"><span>Dry-Spell Duration</span><span class="ctrl-val" id="dry-days-val">10 days</span></div>
        <input type="range" id="dry-days-slider" min="1" max="12" value="10" oninput="updateSequence()">
      </div>
      <div class="ctrl-group">
        <div class="ctrl-label"><span>Peak Temperature (°C)</span><span class="ctrl-val" id="init-temp-val">36.0 °C</span></div>
        <input type="range" id="init-temp-slider" min="20" max="45" value="36" step="0.5" oninput="updateSequence()">
      </div>

      <div class="sec-title" style="margin-top:1.25rem">Day-by-Day Risk Heatmap</div>
      <div class="heatmap-row" id="lstm-heatmap"></div>

      <div class="sec-title">Daily Breakdown</div>
      <div class="day-cards" id="lstm-day-cards"></div>
    </div>

    <div class="card">
      <div class="card-hdr"><div class="card-title">LSTM Risk Progression</div></div>
      <div class="chart-box-tall">
        <canvas id="seq-chart"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════ METRICS TAB ═══════════════ -->
<div id="tab-metrics" class="tab-content">

  <div class="hero-band">
    <div>
      <div class="hero-badge">Benchmark Evaluation</div>
      <h2>Quantitative Model Performance</h2>
      <p>Full comparison of Feature-Attention NN against Logistic Regression, Random Forest, Baseline MLP, and LSTM sequence models across accuracy, precision, recall, F1, and AUC-ROC.</p>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:1.5rem">
    <div class="card">
      <div class="card-hdr"><div class="card-title">Wildfire — Model Comparison</div></div>
      <div class="chart-box"><canvas id="w-bar-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-hdr"><div class="card-title">Flood — Model Comparison</div></div>
      <div class="chart-box"><canvas id="f-bar-chart"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:1.5rem">
    <div class="card-hdr"><div class="card-title">Wildfire Hazard Classification Performance</div></div>
    <div class="tbl-wrap"><table id="table-wildfire">
      <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC-ROC</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </div>

  <div class="card" style="margin-bottom:1.5rem">
    <div class="card-hdr"><div class="card-title">Flood Hazard Classification Performance</div></div>
    <div class="tbl-wrap"><table id="table-flood">
      <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC-ROC</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </div>

  <div class="card">
    <div class="card-hdr"><div class="card-title">Sequential Forecasting — LSTM vs Baselines</div></div>
    <div class="tbl-wrap"><table id="table-sequence">
      <thead><tr><th>Model</th><th>Input Context</th><th>Accuracy</th><th>F1</th><th>AUC-ROC</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </div>
</div>

</main>

<footer>DisasterAI · Team 56 · Feature-Attention Neural Network + LSTM · Deep Learning Project 2024</footer>
</div>

<!-- ════════════════ SCRIPTS ════════════════ -->
<script>
// ── PARTICLE SYSTEM ──────────────────────────────────────────────
(function(){
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [], activeTab = 'wildfire';

  function resize(){ W = canvas.width = innerWidth; H = canvas.height = innerHeight; }
  window.addEventListener('resize', resize); resize();

  class Particle {
    constructor(type){
      this.type = type;
      this.reset();
    }
    reset(){
      if(this.type === 'fire'){
        this.x = Math.random()*W;
        this.y = H + 10;
        this.vx = (Math.random()-.5)*0.6;
        this.vy = -(Math.random()*1.2+0.5);
        this.life = 1; this.decay = Math.random()*.008+.004;
        this.r = Math.random()*3+1;
        this.hue = Math.random()*30+10;
      } else {
        this.x = Math.random()*W;
        this.y = -10;
        this.vx = (Math.random()-.5)*0.3;
        this.vy = Math.random()*1.5+0.8;
        this.life = 1; this.decay = Math.random()*.006+.003;
        this.r = Math.random()*1.5+0.5;
      }
    }
    update(){
      this.x += this.vx; this.y += this.vy;
      this.life -= this.decay;
      if(this.life <= 0 || this.y < -20 || this.y > H+20) this.reset();
    }
    draw(){
      ctx.save();
      ctx.globalAlpha = Math.max(0, this.life) * 0.6;
      if(this.type === 'fire'){
        const g = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.r*2);
        g.addColorStop(0, `hsl(${this.hue},100%,80%)`);
        g.addColorStop(1, `hsla(${this.hue},100%,40%,0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(this.x, this.y, this.r*2, 0, Math.PI*2);
        ctx.fill();
      } else {
        ctx.strokeStyle = 'rgba(100,180,255,0.7)';
        ctx.lineWidth = this.r * 0.5;
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.x + this.vx*4, this.y + this.vy*4);
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  function initParticles(type){
    particles = [];
    for(let i=0;i<80;i++) particles.push(new Particle(type));
  }

  function loop(){
    ctx.clearRect(0,0,W,H);
    particles.forEach(p=>{ p.update(); p.draw(); });
    requestAnimationFrame(loop);
  }

  window.setParticleMode = (mode) => initParticles(mode);
  initParticles('fire');
  loop();
})();

// ── PARAMS ──────────────────────────────────────────────────────
const w_params = [
  {id:"Temperature",name:"Temperature (°C)",min:15,max:45,val:34,step:0.5},
  {id:"RH",name:"Relative Humidity (%)",min:10,max:100,val:45,step:1},
  {id:"Ws",name:"Wind Speed (km/h)",min:2,max:40,val:18,step:1},
  {id:"Rain",name:"Rainfall (mm)",min:0,max:20,val:0,step:0.1},
  {id:"FFMC",name:"Fine Fuel Moisture Code",min:30,max:99,val:85,step:1},
  {id:"DMC",name:"Duff Moisture Code",min:1,max:70,val:25,step:1},
  {id:"DC",name:"Drought Code",min:5,max:250,val:110,step:1},
  {id:"ISI",name:"Initial Spread Index",min:0,max:25,val:9.5,step:0.5},
  {id:"BUI",name:"Build Up Index",min:1,max:80,val:30,step:1},
  {id:"FWI",name:"Fire Weather Index",min:0,max:40,val:14,step:0.5}
];
const f_params = [
  {id:"MonsoonIntensity",name:"Monsoon Intensity",min:0,max:10,val:7.5,step:0.1},
  {id:"TopographyDrainage",name:"Topography Drainage",min:0,max:10,val:4.0,step:0.1},
  {id:"RiverManagement",name:"River Management",min:0,max:10,val:3.5,step:0.1},
  {id:"Deforestation",name:"Deforestation Index",min:0,max:10,val:6.0,step:0.1},
  {id:"Urbanization",name:"Urbanization Rate",min:0,max:10,val:7.0,step:0.1},
  {id:"ClimateChange",name:"Climate Vulnerability",min:0,max:10,val:6.5,step:0.1},
  {id:"DamsQuality",name:"Dams Quality Score",min:0,max:10,val:4.0,step:0.1},
  {id:"Siltation",name:"Siltation Level",min:0,max:10,val:6.0,step:0.1},
  {id:"DrainageSystems",name:"Drainage Systems",min:0,max:10,val:3.5,step:0.1},
  {id:"Rainfall",name:"24h Rainfall (mm)",min:10,max:300,val:180,step:1}
];

let seq_chart = null;

// ── INIT CONTROLS ────────────────────────────────────────────────
function initControls(){
  const render = (containerId, params, updateFn) => {
    document.getElementById(containerId).innerHTML = params.map(p => `
      <div class="ctrl-group">
        <div class="ctrl-label"><span>${p.name}</span><span class="ctrl-val" id="val-${p.id}">${p.val}</span></div>
        <input type="range" id="input-${p.id}" min="${p.min}" max="${p.max}" value="${p.val}" step="${p.step}" oninput="${updateFn}()">
      </div>`).join('');
  };
  render('wildfire-controls', w_params, 'updateWildfire');
  render('flood-controls', f_params, 'updateFlood');
}

// ── GAUGE HELPERS ────────────────────────────────────────────────
function setGauge(arcId, needleId, pctId, catId, pct, color, category){
  const arc = document.getElementById(arcId);
  const needle = document.getElementById(needleId);
  const pctEl = document.getElementById(pctId);
  const catEl = document.getElementById(catId);
  const total = 251.2;
  const offset = total - (pct / 100) * total;
  arc.style.strokeDashoffset = offset;
  const angleDeg = -90 + (pct / 100) * 180;
  needle.style.transform = `rotate(${angleDeg}deg)`;
  pctEl.textContent = pct + '%';
  pctEl.style.color = color;
  catEl.textContent = category;
  catEl.style.background = color + '22';
  catEl.style.color = color;
  catEl.style.border = '1px solid ' + color + '44';
}

// ── ATTENTION BARS ────────────────────────────────────────────────
function renderAttBars(containerId, weights, color){
  const vals = Object.values(weights);
  const maxW = Math.max(...vals);
  document.getElementById(containerId).innerHTML = Object.entries(weights).map(([k,v]) => `
    <div class="risk-meter-bar-wrap">
      <div class="risk-meter-label" title="${k}">${k}</div>
      <div class="risk-meter-track">
        <div class="risk-meter-fill" style="width:${(v/maxW*100).toFixed(1)}%;background:${color}"></div>
      </div>
      <div class="risk-meter-pct">${(v*100).toFixed(1)}%</div>
    </div>`).join('');
}

// ── ALERT BANNER ─────────────────────────────────────────────────
function showAlert(type, risk_pct, category, color){
  const banner = document.getElementById('alert-banner');
  if(risk_pct >= 55){
    banner.style.display = 'block';
    banner.style.background = color + '12';
    banner.style.borderColor = color;
    document.getElementById('alert-icon').textContent = risk_pct >= 80 ? '🚨' : '⚠️';
    document.getElementById('alert-title').textContent = `${category} — ${type} Hazard Detected (${risk_pct}%)`;
    document.getElementById('alert-desc').textContent = risk_pct >= 80
      ? 'Critical threshold exceeded. Immediate attention recommended.'
      : 'Elevated risk detected. Monitor conditions closely.';
    document.getElementById('alert-title').style.color = color;
  } else {
    banner.style.display = 'none';
  }
}

// ── WILDFIRE ─────────────────────────────────────────────────────
async function updateWildfire(){
  const payload = {};
  w_params.forEach(p => {
    const v = parseFloat(document.getElementById(`input-${p.id}`).value);
    document.getElementById(`val-${p.id}`).textContent = v;
    payload[p.id] = v;
  });
  const res = await fetch('/api/predict/wildfire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d = await res.json();
  setGauge('w-gauge-arc','w-needle','w-gauge-pct','w-gauge-cat', d.risk_percentage, d.color, d.category);
  document.getElementById('w-kpi-risk').textContent = d.risk_percentage + '%';
  document.getElementById('w-kpi-risk').style.color = d.color;
  document.getElementById('w-kpi-cat').textContent = d.category;
  const topDriver = Object.entries(d.attention_weights).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('w-kpi-driver').textContent = topDriver[0];
  renderAttBars('w-att-bars', d.attention_weights, d.color);
  showAlert('Wildfire', d.risk_percentage, d.category, d.color);
}

// ── FLOOD ─────────────────────────────────────────────────────────
async function updateFlood(){
  const payload = {};
  f_params.forEach(p => {
    const v = parseFloat(document.getElementById(`input-${p.id}`).value);
    document.getElementById(`val-${p.id}`).textContent = v;
    payload[p.id] = v;
  });
  const res = await fetch('/api/predict/flood',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d = await res.json();
  setGauge('f-gauge-arc','f-needle','f-gauge-pct','f-gauge-cat', d.risk_percentage, d.color, d.category);
  document.getElementById('f-kpi-risk').textContent = d.risk_percentage + '%';
  document.getElementById('f-kpi-risk').style.color = d.color;
  document.getElementById('f-kpi-cat').textContent = d.category;
  const topDriver = Object.entries(d.attention_weights).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('f-kpi-driver').textContent = topDriver[0];
  renderAttBars('f-att-bars', d.attention_weights, d.color);
  showAlert('Flood', d.risk_percentage, d.category, d.color);
}

// ── SEQUENCE ─────────────────────────────────────────────────────
async function updateSequence(){
  const days = parseInt(document.getElementById('dry-days-slider').value);
  const temp = parseFloat(document.getElementById('init-temp-slider').value);
  document.getElementById('dry-days-val').textContent = days + ' days';
  document.getElementById('init-temp-val').textContent = temp.toFixed(1) + ' °C';

  const res = await fetch('/api/predict/sequence',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dry_spell_days:days,init_temp:temp})});
  const d = await res.json();
  const probs = d.probabilities;

  // KPI
  const peak = Math.max(...probs);
  const first = probs[0];
  const trend = peak > first ? '↑ Increasing' : peak < first ? '↓ Decreasing' : '→ Stable';
  document.getElementById('lstm-kpi-peak').textContent = peak + '%';
  document.getElementById('lstm-kpi-trend').textContent = trend;

  // Heatmap
  const hmap = document.getElementById('lstm-heatmap');
  hmap.innerHTML = probs.map((p,i) => {
    const c = probToColor(p);
    return `<div class="heat-cell" style="background:${c};box-shadow:0 0 10px ${c}55" title="Day ${i+1}: ${p}%">
      <span style="font-size:.75rem;font-weight:800">${p}%</span>
      <span class="day-lbl">D${i+1}</span>
    </div>`;
  }).join('');

  // Day cards
  const dcont = document.getElementById('lstm-day-cards');
  dcont.innerHTML = probs.map((p,i) => {
    const c = probToColor(p);
    return `<div class="day-card">
      <div class="dc-label">Day ${i+1}</div>
      <div class="dc-pct" style="color:${c}">${p}%</div>
      <div class="dc-bar" style="background:${c};width:${p}%"></div>
    </div>`;
  }).join('');

  // Line chart
  const ctx = document.getElementById('seq-chart').getContext('2d');
  if(seq_chart) seq_chart.destroy();

  const gradient = ctx.createLinearGradient(0,0,0,300);
  gradient.addColorStop(0,'rgba(255,69,0,0.3)');
  gradient.addColorStop(1,'rgba(255,69,0,0.0)');

  seq_chart = new Chart(ctx, {
    type:'line',
    data:{
      labels: d.days,
      datasets:[{
        label:'LSTM Fire Probability (%)',
        data: probs,
        borderColor:'#FF4500',
        backgroundColor: gradient,
        fill:true,
        tension:0.4,
        pointRadius:6,
        pointBackgroundColor: probs.map(p => probToColor(p)),
        pointBorderColor:'#fff',
        pointBorderWidth:2,
        pointHoverRadius:9
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      scales:{
        y:{min:0,max:100,
          grid:{color:'rgba(255,255,255,.04)'},
          ticks:{color:'#607B96',callback:v=>v+'%'},
          border:{color:'transparent'}
        },
        x:{
          grid:{color:'rgba(255,255,255,.04)'},
          ticks:{color:'#607B96'},
          border:{color:'transparent'}
        }
      },
      plugins:{
        legend:{labels:{color:'#E8EDF5',font:{weight:'600'}}},
        tooltip:{
          backgroundColor:'rgba(8,16,32,.95)',
          borderColor:'rgba(255,255,255,.1)',
          borderWidth:1,
          titleColor:'#E8EDF5',
          bodyColor:'#607B96',
          callbacks:{label:ctx=>' Fire Risk: '+ctx.parsed.y+'%'}
        }
      }
    }
  });
}

function probToColor(p){
  if(p < 25) return '#00E676';
  if(p < 50) return '#FFD600';
  if(p < 75) return '#FF6D00';
  return '#FF1744';
}

// ── METRICS ──────────────────────────────────────────────────────
let wBarChart, fBarChart;
async function loadMetrics(){
  const res = await fetch('/api/metrics');
  const data = await res.json();

  const renderTable = (id, rows, isSeq=false) => {
    document.querySelector(`#${id} tbody`).innerHTML = rows.map(r => {
      const isOurs = r.model.includes('Ours') || r.model.includes('LSTM (Full');
      return `<tr${isOurs?' class="ours"':''}>
        <td><strong>${r.model}</strong>${isOurs?'<span class="badge-ours">OUR MODEL</span>':''}</td>
        ${isSeq ? `<td style="color:var(--muted);font-size:.78rem">${r.input}</td>` : `<td>${r.precision}</td><td>${r.recall}</td>`}
        <td class="${r.accuracy>0.92?'metric-good':''}">${(r.accuracy*100).toFixed(1)}%</td>
        <td class="${r.f1>0.92?'metric-good':''}">${(r.f1*100).toFixed(1)}%</td>
        <td class="${r.auc>0.98?'metric-best':'metric-good'}">${r.auc}</td>
      </tr>`;
    }).join('');
  };
  renderTable('table-wildfire', data.wildfire);
  renderTable('table-flood', data.flood);
  renderTable('table-sequence', data.sequence, true);

  // Bar charts
  const barOpts = (label, color) => ({
    responsive:true, maintainAspectRatio:false,
    indexAxis:'y',
    scales:{
      x:{min:0.8,max:1,grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#607B96',callback:v=>(v*100).toFixed(0)+'%'},border:{color:'transparent'}},
      y:{grid:{display:false},ticks:{color:'#E8EDF5',font:{size:10}},border:{color:'transparent'}}
    },
    plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+(c.parsed.x*100).toFixed(1)+'%'}}}
  });

  const buildBar = (id, rows, metric, color) => {
    const ctx = document.getElementById(id).getContext('2d');
    const colors = rows.map((r,i)=> i===rows.length-1 ? color : 'rgba(255,255,255,.1)');
    return new Chart(ctx, {
      type:'bar',
      data:{
        labels: rows.map(r=>r.model.replace(' (Ours)','')),
        datasets:[{label:'AUC-ROC',data:rows.map(r=>r[metric]),backgroundColor:colors,borderRadius:6}]
      },
      options: barOpts('AUC-ROC', color)
    });
  };

  if(wBarChart) wBarChart.destroy();
  if(fBarChart) fBarChart.destroy();
  wBarChart = buildBar('w-bar-chart', data.wildfire, 'auc', 'rgba(255,69,0,.8)');
  fBarChart  = buildBar('f-bar-chart', data.flood, 'auc', 'rgba(0,119,255,.8)');
}

// ── TAB SWITCH ────────────────────────────────────────────────────
function switchTab(id, evt){
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  if(evt) evt.target.classList.add('active');
  document.getElementById('tab-'+id).classList.add('active');
  document.getElementById('alert-banner').style.display = 'none';
  if(id==='lstm'){ setParticleMode('fire'); updateSequence(); }
  else if(id==='flood'){ setParticleMode('rain'); updateFlood(); }
  else if(id==='wildfire'){ setParticleMode('fire'); updateWildfire(); }
  else if(id==='metrics'){ setParticleMode('fire'); loadMetrics(); }
}

// ── BOOT ─────────────────────────────────────────────────────────
window.onload = () => {
  initControls();
  updateWildfire();
  updateFlood();
};
</script>
</body>
</html>
"""
