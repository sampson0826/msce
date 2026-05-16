#!/usr/bin/env python3
"""
ExecutorHunter Visual Demo — 直观展示不可观测执行者的搜索过程

Run: cd /Users/dengxinhang/paper && python3 constraint_residual/demo.py && open constraint_residual/demo_output.html
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/dengxinhang/paper')
from constraint_residual.executor_models import Executor
from constraint_residual.executor_hunter import ExecutorHunter
from constraint_residual.experiment_designer import ExperimentDesigner
from constraint_residual.physics_executors import (
    build_known_executors, build_missing_executor_gaps,
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output.html")

# ---- Data generation ----

def make_executor(id, name, fn):
    return Executor(id=id, name=name, from_layer=0, to_layer=1,
                    executor_type='E-I', certainty=1.0, transmission_fn=fn)

# Executor A: strong constraint at x ≈ 0.3
eA = make_executor('E_A', '规范场强制',
    lambda uc, p: np.array([uc[0]*np.exp(-((p[0]-0.3)/0.3)**2),
                             uc[1]*np.exp(-((p[1]-0.3)/0.3)**2)]))
# Executor B: strong constraint at x ≈ 0.7
eB = make_executor('E_B', '守恒律强制',
    lambda uc, p: np.array([uc[0]*np.exp(-((p[0]-0.7)/0.3)**2),
                             uc[1]*np.exp(-((p[1]-0.7)/0.3)**2)]))

# Executor C: opposes A → creates dark zone at x ≈ 0.5, y ≈ 0.5
eC = make_executor('E_C', '反向约束',
    lambda uc, p: np.array([-uc[0]*np.exp(-((p[0]-0.5)/0.2)**2),
                             -uc[1]*np.exp(-((p[1]-0.5)/0.2)**2)]))

# Executor D: another constraint creating overlapping zones
eD = make_executor('E_D', '泡利不相容强制',
    lambda uc, p: np.array([uc[0]*np.exp(-((p[0]-0.5)/0.4)**2)*np.sin(p[1]*3),
                             uc[1]*np.exp(-((p[1]-0.5)/0.4)**2)*np.cos(p[0]*3)]))

# Full system
all_executors = [eA, eB, eC, eD]

# Hunt with all 4 executors
hunter = ExecutorHunter(all_executors, residual_epsilon=0.05, dark_zone_cancellation_eps=0.1)
bounds = [(0, 1), (0, 1)]
gaps = hunter.hunt_gaps(from_layer=0, to_layer=1, bounds=bounds, n_points=50)

# Transmission completeness
trans = hunter.compute_transmission_completeness(all_executors, bounds, n_points=50)

# Also hunt with only 2 executors to show how gaps increase
hunter_partial = ExecutorHunter([eA, eB], residual_epsilon=0.05, dark_zone_cancellation_eps=0.1)
gaps_partial = hunter_partial.hunt_gaps(from_layer=0, to_layer=1, bounds=bounds, n_points=50)

# Dark zone scan
dark_clusters = hunter.dark_detector.scan(
    hunter._build_constraint_field(all_executors), bounds, n_points=50
)

# Build 2D field for visualization
field = hunter._build_constraint_field(all_executors)
xs = np.linspace(0, 1, 60)
ys = np.linspace(0, 1, 60)
heatmap = np.zeros((60, 60))
cancellation_map = np.zeros((60, 60))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        p = np.array([x, y])
        grad = field.constraint_gradient(p)
        heatmap[j, i] = float(np.linalg.norm(grad))
        # Individual magnitudes
        ind_mags = [float(np.linalg.norm(field.rules[k].gradient(p))) for k in range(len(field.rules))]
        total_ind = sum(ind_mags)
        total_combined = float(np.linalg.norm(grad))
        if total_ind > 1e-10:
            cancellation_map[j, i] = total_combined / total_ind
        else:
            cancellation_map[j, i] = 1.0

# Design experiments
designer = ExperimentDesigner(S0=1.0)
proposals = designer.design_for_gaps(gaps)
ranked = designer.top_experiments(proposals, n=5)

# Physics executors summary
phys_execs = build_known_executors()
phys_gaps = build_missing_executor_gaps()

# ---- Generate HTML ----

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ExecutorHunter — 不可观测执行者搜索</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a14; color:#d0d0e0; font-family:'SF Mono','Menlo',monospace; padding:20px 40px; }}
h1 {{ font-size:22px; color:#ffffff; margin-bottom:4px; }}
h2 {{ font-size:15px; color:#8899cc; margin:28px 0 10px; border-bottom:1px solid #1a1a3a; padding-bottom:4px; }}
.sub {{ color:#667; font-size:12px; margin-bottom:30px; }}
.row {{ display:flex; gap:20px; flex-wrap:wrap; }}
.card {{ background:#0f0f22; border:1px solid #1a1a3a; border-radius:8px; padding:16px; flex:1; min-width:340px; }}
.card h3 {{ font-size:13px; color:#aabbdd; margin-bottom:10px; }}
.chart-wrap {{ position:relative; height:280px; }}
.metric {{ display:inline-block; background:#111133; border-radius:6px; padding:10px 16px; margin:4px; text-align:center; }}
.metric .val {{ font-size:24px; font-weight:bold; color:#88ddff; }}
.metric .lbl {{ font-size:10px; color:#667; }}
.gap-row {{ display:flex; align-items:center; padding:8px 12px; margin:4px 0; background:#111122; border-radius:4px; border-left:3px solid; }}
.gap-row .badge {{ font-size:9px; padding:2px 6px; border-radius:3px; margin-right:8px; }}
.badge.ei {{ background:#225533; color:#44dd66; }}
.badge.eii {{ background:#334422; color:#ddcc44; }}
.badge.eiii {{ background:#442222; color:#dd6644; }}
.badge.dz {{ background:#331144; color:#cc44dd; }}
.priority-bar {{ height:4px; background:#223; border-radius:2px; margin-top:4px; }}
.priority-fill {{ height:100%; border-radius:2px; }}
.exp-card {{ background:#0f0f22; border:1px solid #1a1a3a; border-radius:6px; padding:12px; margin:8px 0; }}
.exp-card .type-tag {{ font-size:9px; padding:2px 6px; border-radius:3px; }}
.type-extreme {{ background:#441122; color:#ff6688; }}
.type-precision {{ background:#112244; color:#66aaff; }}
.type-symbreak {{ background:#331144; color:#cc66ff; }}
.type-interference {{ background:#114422; color:#66dd88; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; margin:8px 0; }}
th {{ background:#111133; color:#889; padding:6px 10px; text-align:left; }}
td {{ padding:6px 10px; border-bottom:1px solid #111122; }}
td:first-child {{ color:#aabbdd; }}
.ev-high {{ color:#44dd66; }}
.ev-medium {{ color:#ddcc44; }}
.ev-low {{ color:#dd6644; }}
footer {{ color:#444; font-size:10px; text-align:center; margin-top:40px; padding:20px; }}
</style>
</head>
<body>

<h1>ExecutorHunter</h1>
<div class="sub">不可观测执行者搜索模型 · 可视化演示<br>
核心命题：越稳固的规则，越不可见。约束残差法让不可见变为可计算。</div>

<!-- METRICS -->
<h2>系统概览</h2>
<div class="row">
<div class="metric"><div class="val">{len(all_executors)}</div><div class="lbl">已知执行者</div></div>
<div class="metric"><div class="val">{len(gaps)}</div><div class="lbl">检测到缺口</div></div>
<div class="metric"><div class="val">{len(dark_clusters)}</div><div class="lbl">暗区簇</div></div>
<div class="metric"><div class="val">{trans.completeness_mean:.1%}</div><div class="lbl">平均传输完整性</div></div>
<div class="metric"><div class="val">{len(ranked)}</div><div class="lbl">实验提案</div></div>
</div>

<!-- CONSTRAINT FIELD -->
<h2>约束力场 ∥Π(p)∥ — 状态空间中的约束强度</h2>
<div class="row">
<div class="card" style="flex:1.2">
<h3>总约束力热力图</h3>
<div class="chart-wrap"><canvas id="heatmapChart"></canvas></div>
</div>
<div class="card">
<h3>解读</h3>
<div style="font-size:11px;color:#889;line-height:1.7">
<p>热力图中<b style="color:#ffdd88">亮色区域</b> = 约束力强（已知规则在此处活跃）</p>
<p><b style="color:#4444dd">暗色区域</b> = 约束力弱 → <b style="color:#ff6644">可能存在未知规则</b></p>
<p>如果在亮区但合约束力为零 → <b style="color:#cc44dd">Type III 暗区</b></p>
<p style="margin-top:8px;color:#667">4个执行者（规范场+守恒律+反向约束+泡利不相容）在2D状态空间中产生交叉约束</p>
</div>
</div>
</div>

<!-- CANCELLATION MAP -->
<h2>暗区检测 — 取消率 ∥Σ∇σ∥ / Σ∥∇σ∥</h2>
<div class="row">
<div class="card" style="flex:1.2">
<h3>取消率热力图（蓝色=暗区）</h3>
<div class="chart-wrap"><canvas id="cancelChart"></canvas></div>
</div>
<div class="card">
<h3>暗区分析</h3>
<div style="font-size:11px;color:#889;line-height:1.7">"""
for dc in dark_clusters:
    html += f"""<p style="margin:4px 0;padding:6px;background:#111122;border-radius:4px">
<b style="color:#cc44dd">{dc.balance_topology}</b> ·
{len(dc.points)}点 · 平均取消率={dc.mean_cancellation_ratio:.4f} ·
涉及 {', '.join(dc.constraints_involved[:3])}</p>"""
if not dark_clusters:
    html += """<p style="color:#667">当前配置下未检测到暗区 — 这是好消息，意味着没有未知规则被完美隐藏。但换个参数区域可能存在。</p>"""
html += """</div></div></div>"""

# EXECUTOR GAPS
html += f"""<h2>执行者缺口 · 搜索 L0→L1</h2>
<div class="card"><table>
<tr><th>优先级</th><th>位置</th><th>传输完整性</th><th>残差</th><th>候选类型</th><th>数学形式预测</th></tr>"""
for g in gaps[:8]:
    ttag = {'E-I':'ei','E-II':'eii','E-III':'eiii'}.get(g.candidate_type,'')
    dz_tag = ' <span class="badge dz">暗区</span>' if g.dark_zone_ids else ''
    html += f"""<tr>
<td>{g.priority:.2f}</td>
<td>({g.region[0]:.2f}, {g.region[1]:.2f})</td>
<td style="color:{'#44dd66' if g.transmission_completeness>0.8 else '#dd6644'}">{g.transmission_completeness:.1%}</td>
<td style="color:{'#44dd66' if g.residual_magnitude<0.1 else '#dd6644'}">{g.residual_magnitude:.3f}</td>
<td><span class="badge {ttag}">{g.candidate_type}{dz_tag}</span></td>
<td style="font-size:10px">{g.candidate_math_form[:80] if g.candidate_math_form else '—'}</td></tr>"""
html += """</table></div>"""

# PARTIAL SYSTEM COMPARISON
html += f"""<h2>对比：减少执行者 → 缺口增多</h2>
<div class="row">
<div class="metric"><div class="val">{len(all_executors)}→{len(gaps)}</div><div class="lbl">4个执行者 → 缺口</div></div>
<div class="metric"><div class="val">2→{len(gaps_partial)}</div><div class="lbl">2个执行者 → 缺口</div></div>
<div class="metric"><div class="val">+{len(gaps_partial)-len(gaps)}</div><div class="lbl">缺口增量</div></div>
</div>
<p style="color:#667;font-size:11px;margin-top:8px">
移除执行者C和D后，缺口从 {len(gaps)} 个增加到 {len(gaps_partial)} 个——证明系统能检测到"执行者缺失"。</p>"""

# EXPERIMENTS
html += f"""<h2>实验提案 · 暴露隐藏执行者</h2>"""
for i,p in enumerate(ranked):
    tcls = {'extreme_energy':'type-extreme','precision':'type-precision',
            'symmetry_breaking':'type-symbreak','interference':'type-interference'}
    html += f"""<div class="exp-card">
<div style="display:flex;justify-content:space-between;align-items:center">
<strong>#{i+1} {p.id}</strong>
<div>
<span class="type-tag {tcls.get(p.experiment_type,'')}">{p.experiment_type}</span>
<span style="font-size:10px;color:#667;margin-left:8px">优先度={p.priority_score:.3f}</span>
</div>
</div>
<div style="display:flex;gap:16px;margin:8px 0;font-size:11px">
<div>可行性: <b style="color:{'#44dd66' if p.feasibility>0.5 else '#ddcc44' if p.feasibility>0.2 else '#dd6644'}">{p.feasibility:.0%}</b></div>
<div>发现潜力: <b style="color:#88ddff">{p.discovery_potential:.0%}</b></div>
<div>预测信号: <b style="color:#ffdd88">{p.predicted_signal_strength:.4f}</b></div>
<div>所需精度: <b>{p.required_precision:.3e}</b></div>
</div>
<p style="font-size:10px;color:#889">{p.rationale}</p>
</div>"""

# PHYSICS MAP
html += f"""<h2>物理执行者拓扑 · 已知 + 缺失</h2>
<div class="card"><table>
<tr><th>ID</th><th>名称</th><th>类型</th><th>层级</th><th>确定性</th><th>证据</th></tr>"""
for e in phys_execs:
    evc = 'ev-high' if e.evidence_strength=='high' else 'ev-medium'
    html += f"""<tr><td>{e.id}</td><td>{e.name}</td>
<td><span class="badge ei">{e.executor_type}</span></td>
<td>L{e.from_layer}→L{e.to_layer}</td><td>{e.certainty:.0%}</td>
<td class="{evc}">{e.evidence_strength}</td></tr>"""
html += """<tr style="background:#221122"><td colspan="6" style="color:#cc44dd;text-align:center">
<b>▼ 缺失执行者（约束残差法检测）</b></td></tr>"""
for i,g in enumerate(phys_gaps):
    html += f"""<tr style="background:#110f18">
<td style="color:#cc44dd">M{i+1}</td>
<td style="color:#cc88dd">{g.candidate_math_form[:50]}...</td>
<td><span class="badge {'ei' if g.candidate_type=='E-I' else 'eii'}">{g.candidate_type}?</span></td>
<td>L{g.from_layer}→L{g.to_layer}</td>
<td>{g.candidate_type_confidence:.0%}</td>
<td class="ev-low">speculative</td></tr>"""
html += """</table></div>"""

# JS CHARTS
html += f"""
<script>
const heatData = {json.dumps(heatmap.tolist())};
const cancelData = {json.dumps(cancellation_map.tolist())};
const xs = {json.dumps(xs.tolist())};
const ys = {json.dumps(ys.tolist())};

function makeHeatmap(ctx, data, label, colorScale) {{
    const labels = xs.map(x => x.toFixed(2));
    const yLabels = ys.map(y => y.toFixed(2));
    const datasets = [];
    for (let i = 0; i < data.length; i++) {{
        datasets.push({{
            label: i === 0 ? 'Row 0' : '',
            data: data[i],
            backgroundColor: data[i].map(v => {{
                const t = Math.min(v / (colorScale || 2.0), 1.0);
                return `rgba(${{Math.floor(t*255)}},${{Math.floor(t*128)}},${{Math.floor((1-t)*200)}},0.85)`;
            }}),
            borderWidth: 0,
        }});
    }}
    new Chart(ctx, {{
        type: 'bar',
        data: {{ labels: yLabels, datasets: datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => 'value: '+ctx.raw.toFixed(3) }} }} }},
            scales: {{
                x: {{ stacked: true, display: false }},
                y: {{ stacked: true, display: false }}
            }}
        }}
    }});
}}

// Use a matrix-style heatmap via stacked bar chart
// Actually let's use a simpler approach: imageData-style via stacked bars
makeHeatmap(document.getElementById('heatmapChart'), heatData, 'Constraint Force', 2.5);
makeHeatmap(document.getElementById('cancelChart'), cancelData, 'Cancellation Ratio', 1.05);
</script>

<footer>
ExecutorHunter · 规则认知体系 · 约束残差法 · 2026<br>
"越稳固的规则，越不可见" — 但不可见不等于不可发现
</footer>
</body></html>"""

with open(OUT_PATH, 'w') as f:
    f.write(html)
print(f"Demo written to {OUT_PATH}")
print(f"Transmission: {trans.completeness_mean:.1%}")
print(f"Gaps found: {len(gaps)}")
print(f"Dark zones: {len(dark_clusters)}")
print(f"Experiment proposals: {len(ranked)}")
