#!/usr/bin/env python3
"""Generate publication-quality SVG figures for the uEngram research paper."""
from pathlib import Path

FIGURES_DIR = Path("docs/research-paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def create_figure1_system_architecture():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 280" width="100%" height="100%">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
    </marker>
    <style>
      .box { fill: #f8fafc; stroke: #94a3b8; stroke-width: 1.5; rx: 6px; }
      .box-fast { fill: #eff6ff; stroke: #3b82f6; stroke-width: 1.5; rx: 6px; }
      .box-offline { fill: #fdf4ff; stroke: #a855f7; stroke-width: 1.5; rx: 6px; }
      .title { font-family: Inter, system-ui, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
      .sub { font-family: Inter, system-ui, sans-serif; font-size: 11px; fill: #475569; }
      .path-lbl { font-family: Inter, system-ui, sans-serif; font-size: 11px; font-weight: 600; fill: #2563eb; }
      .line { stroke: #64748b; stroke-width: 1.5; fill: none; marker-end: url(#arrow); }
      .line-dashed { stroke: #a855f7; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; marker-end: url(#arrow); }
    </style>
  </defs>

  <!-- Section Labels -->
  <text x="20" y="24" class="path-lbl">FAST DETERMINISTIC EDGE PATH (&lt; 0.1 ms)</text>
  <text x="20" y="200" class="path-lbl" style="fill: #9333ea;">OFFLINE COMMISSIONING PATH (Server / Host)</text>

  <!-- Fast Path Boxes -->
  <rect x="20" y="40" width="150" height="90" class="box-fast" />
  <text x="35" y="68" class="title">Causal Sensor Window</text>
  <text x="35" y="88" class="sub">15-sec Telemetry X</text>
  <text x="35" y="106" class="sub">Pre-fit SVD Diagnostics</text>

  <path d="M 170 85 L 210 85" class="line" />

  <rect x="210" y="40" width="160" height="90" class="box-fast" />
  <text x="225" y="68" class="title">Edge Whitening</text>
  <text x="225" y="88" class="sub">DSP / 46KB CNN / Roles</text>
  <text x="225" y="106" class="sub">PCA + MAD Noise Cov N</text>

  <path d="M 370 85 L 410 85" class="line" />

  <rect x="410" y="40" width="170" height="90" class="box-fast" />
  <text x="425" y="68" class="title">μEngram Bounded Slots</text>
  <text x="425" y="88" class="sub">Semantic (EMA)</text>
  <text x="425" y="106" class="sub">Candidate | Episodic (Rarity)</text>

  <path d="M 580 85 L 620 85" class="line" />

  <rect x="620" y="40" width="160" height="90" class="box-fast" />
  <text x="635" y="68" class="title">EdgeMemory Lite DB</text>
  <text x="635" y="88" class="sub">Idempotent Event Ledger</text>
  <text x="635" y="106" class="sub">Pinned Waveform Ring</text>

  <path d="M 780 85 L 810 85" class="line" />

  <rect x="810" y="40" width="95" height="90" class="box" />
  <text x="822" y="75" class="title">Diagnostics</text>
  <text x="822" y="98" class="sub">Post-LLM /</text>
  <text x="822" y="114" class="sub">Operator</text>

  <!-- Offline Distillation Boxes -->
  <rect x="180" y="215" width="220" height="50" class="box-offline" />
  <text x="195" y="237" class="title">Foundation Teachers</text>
  <text x="195" y="253" class="sub">MOMENT-1 (141MB) + Chronos-2 (478MB)</text>

  <path d="M 400 240 L 470 240" class="line-dashed" />

  <rect x="470" y="215" width="230" height="50" class="box-offline" />
  <text x="485" y="237" class="title">Dimension-Normalized Distillation</text>
  <text x="485" y="253" class="sub">Trains 46KB Edge Student S_θ</text>

  <path d="M 585 215 L 585 170 L 290 170 L 290 130" class="line-dashed" />
</svg>"""
    (FIGURES_DIR / "system-architecture.svg").write_text(svg, encoding="utf-8")


def create_figure2_state_machine():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 260" width="100%" height="100%">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/>
    </marker>
    <style>
      .slot-box { fill: #ffffff; stroke: #64748b; stroke-width: 1.5; rx: 6px; }
      .slot-semantic { fill: #f0fdf4; stroke: #16a34a; stroke-width: 1.5; rx: 6px; }
      .slot-candidate { fill: #fefce8; stroke: #ca8a04; stroke-width: 1.5; rx: 6px; }
      .slot-episodic { fill: #fef2f2; stroke: #dc2626; stroke-width: 1.5; rx: 6px; }
      .txt-h { font-family: Inter, system-ui, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
      .txt-p { font-family: Inter, system-ui, sans-serif; font-size: 11px; fill: #334155; }
      .lbl { font-family: Inter, system-ui, sans-serif; font-size: 10px; font-weight: 600; fill: #475569; }
      .flow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow2); }
    </style>
  </defs>

  <!-- Ingestion -->
  <rect x="20" y="90" width="150" height="70" class="slot-box" />
  <text x="35" y="118" class="txt-h">Whitened Vector</text>
  <text x="35" y="138" class="txt-p">z(x) in R^d (MAD whitened)</text>

  <!-- Split paths -->
  <path d="M 170 115 L 260 60" class="flow" />
  <text x="180" y="75" class="lbl">In-Gate (Routine)</text>

  <path d="M 170 135 L 260 190" class="flow" />
  <text x="180" y="180" class="lbl">Out-of-Gate (Novel)</text>

  <!-- Semantic Slots -->
  <rect x="260" y="25" width="230" height="75" class="slot-semantic" />
  <text x="275" y="50" class="txt-h" style="fill: #15803d;">SEMANTIC SLOTS (Routine Baselines)</text>
  <text x="275" y="68" class="txt-p">EMA centroid: μ_t = (1-β)μ_t-1 + βx_t</text>
  <text x="275" y="86" class="txt-p">Updates ONLY if operator-confirmed</text>

  <!-- Candidate Probation -->
  <rect x="260" y="150" width="230" height="75" class="slot-candidate" />
  <text x="275" y="175" class="txt-h" style="fill: #a16207;">CANDIDATE SLOTS (Probation)</text>
  <text x="275" y="193" class="txt-p">Persistence counter (M-of-N windows)</text>
  <text x="275" y="211" class="txt-p">Blocks creeping fault assimilation</text>

  <path d="M 490 60 L 580 60" class="flow" />
  <rect x="580" y="35" width="280" height="55" class="slot-box" />
  <text x="595" y="58" class="txt-h">MEWMA Step Supersession</text>
  <text x="595" y="76" class="txt-p">Marks pre-repair baselines SUPERSEDED</text>

  <path d="M 490 190 L 580 190" class="flow" />
  <text x="500" y="180" class="lbl">Rarity &gt; MinSlot</text>

  <!-- Episodic Slots -->
  <rect x="580" y="145" width="280" height="90" class="slot-episodic" />
  <text x="595" y="170" class="txt-h" style="fill: #b91c1c;">EPISODIC SLOTS (Rare Fault Episodes)</text>
  <text x="595" y="190" class="txt-p">Competitive Rarity Admission vs Weakest Mature</text>
  <text x="595" y="208" class="txt-p">Pins Raw Waveform Slice BEFORE Commit</text>
  <text x="595" y="224" class="txt-p">Protected Quota for Critical Seizure Subtypes</text>
</svg>"""
    (FIGURES_DIR / "slot-state-machine.svg").write_text(svg, encoding="utf-8")


def create_figure4_specialist_performance():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 280" width="100%" height="100%">
  <style>
    .bg-panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 6px; }
    .chart-title { font-family: Inter, system-ui, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
    .bar-lbl { font-family: Inter, system-ui, sans-serif; font-size: 11px; fill: #1e293b; font-weight: 500; }
    .val-lbl { font-family: Inter, system-ui, sans-serif; font-size: 11px; fill: #0f172a; font-weight: 700; }
    .sub-lbl { font-family: Inter, system-ui, sans-serif; font-size: 10px; fill: #64748b; }
    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 2,2; }
  </style>

  <!-- Panel A: OS-Metrics -->
  <rect x="10" y="10" width="440" height="260" class="bg-panel" />
  <text x="30" y="38" class="chart-title">(A) OS-Metrics (5-ch): Strict Recall &amp; False Alarms/Wk</text>

  <!-- Bar 1: DSP Baseline -->
  <text x="30" y="70" class="bar-lbl">Baseline DSP (d=16)</text>
  <rect x="30" y="78" width="220" height="18" fill="#94a3b8" rx="3" />
  <text x="260" y="92" class="val-lbl">64.3% (27/42)</text>
  <text x="30" y="110" class="sub-lbl">False Alarms: 26.6 / week | Fused global vector</text>

  <!-- Bar 2: All-Channel Role -->
  <text x="30" y="135" class="bar-lbl">All-Channel Role (dynamic_health)</text>
  <rect x="30" y="143" width="195" height="18" fill="#cbd5e1" rx="3" />
  <text x="235" y="157" class="val-lbl">57.1% (24/42)</text>
  <text x="30" y="175" class="sub-lbl">False Alarms: 42.0 / week | Diluted across quiet channels</text>

  <!-- Bar 3: Domain Specialists -->
  <text x="30" y="200" class="bar-lbl">Domain Specialists (or_dynamic_health)</text>
  <rect x="30" y="208" width="236" height="18" fill="#2563eb" rx="3" />
  <text x="276" y="222" class="val-lbl" style="fill: #1d4ed8;">69.0% (29/42)</text>
  <text x="30" y="240" class="sub-lbl" style="fill: #1d4ed8; font-weight: 600;">False Alarms: 19.6 / week | 100% Attribution | 6.27 μs/win</text>

  <!-- Panel B: SKAB -->
  <rect x="470" y="10" width="440" height="260" class="bg-panel" />
  <text x="490" y="38" class="chart-title">(B) SKAB Water Loop (8-ch): Strict In-Interval Recall</text>

  <!-- Bar B1: 8-ch CNN -->
  <text x="490" y="70" class="bar-lbl">8-ch Monolithic CNN (All Readouts)</text>
  <rect x="490" y="78" width="20" height="18" fill="#f87171" rx="3" />
  <text x="520" y="92" class="val-lbl" style="fill: #dc2626;">0 / 34 (0.0% Strict)</text>
  <text x="490" y="110" class="sub-lbl">All post-onset warnings, 0% strict fault-interval recall</text>

  <!-- Bar B2: Role dynamic_health -->
  <text x="490" y="135" class="bar-lbl">Role Separation (dynamic_health)</text>
  <rect x="490" y="143" width="140" height="18" fill="#60a5fa" rx="3" />
  <text x="640" y="157" class="val-lbl">8 / 34 (23.5%)</text>
  <text x="490" y="175" class="sub-lbl">False Alarms: 1.59 / hour | Best recall-to-false-alarm trade-off</text>

  <!-- Bar B3: Role at_least_2 -->
  <text x="490" y="200" class="bar-lbl">Role Separation (at_least_2 policy)</text>
  <rect x="490" y="208" width="175" height="18" fill="#16a34a" rx="3" />
  <text x="675" y="222" class="val-lbl" style="fill: #15803d;">10 / 34 (29.4%)</text>
  <text x="490" y="240" class="sub-lbl" style="fill: #15803d; font-weight: 600;">False Alarms: 6.19 / hour | Highest strict recall on SKAB</text>
</svg>"""
    (FIGURES_DIR / "domain-specialist-comparison.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    print("Generating vector SVG figures...")
    create_figure1_system_architecture()
    create_figure2_state_machine()
    create_figure4_specialist_performance()
    print("Figures generated successfully in docs/research-paper/figures/")
