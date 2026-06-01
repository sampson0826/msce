import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']

BG = '#0a0e14'
CARD_BG = '#12171f'
MSCE_C = '#3b82f6'
GPT_C = '#374151'
ACCENT = '#f59e0b'
RED = '#ef4444'
GREEN = '#22c55e'
WHITE = '#f0f3f8'
SUB = '#8b949e'
MUTED = '#545d6b'

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor(BG)

# ===== SECTION 1: Domain Accuracy (top 60%) =====
ax1 = fig.add_axes([0.08, 0.38, 0.84, 0.56])
ax1.set_facecolor(BG)

domains = ['数学', '科学\n(物理)', '跨域\n推理', '逻辑', '约束\n传播', '语言']
msce     = [98.3, 98.9, 86.0, 94.7, 71.2, 93.1]
gpt55    = [95.0, 73.6, 60.5, 89.5, 61.5, 93.1]
gap      = [m - g for m, g in zip(msce, gpt55)]
n_q      = [60, 87, 43, 57, 52, 87]

y = np.arange(len(domains))
h = 0.3

bars_m = ax1.barh(y + h/2, msce, h, color=MSCE_C, edgecolor='none', zorder=3)
bars_g = ax1.barh(y - h/2, gpt55, h, color=GPT_C, edgecolor='none', zorder=3)

for bar, val in zip(bars_m, msce):
    ax1.text(val + 1.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
             va='center', fontsize=12, fontweight='bold', color=MSCE_C)
for bar, val in zip(bars_g, gpt55):
    ax1.text(val + 1.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
             va='center', fontsize=11, color=MUTED)

for i, g in enumerate(gap):
    if g >= 15:
        ax1.text(105, i, f'+{g:.0f}pp', va='center', fontsize=10,
                fontweight='bold', color=ACCENT)
    elif g > 2:
        ax1.text(105, i, f'+{g:.0f}pp', va='center', fontsize=10, color=SUB)
    else:
        ax1.text(105, i, '平', va='center', fontsize=10, color=SUB)

for i, n in enumerate(n_q):
    ax1.text(-1.5, i, f'n={n}', va='center', fontsize=9, color=MUTED, ha='right')

ax1.set_yticks(y)
ax1.set_yticklabels(domains, fontsize=12, color=WHITE)
ax1.set_xlim(0, 118)
ax1.set_xticks([0, 25, 50, 75, 100])
ax1.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color=MUTED)
ax1.xaxis.grid(True, linestyle='--', alpha=0.06, color='white')
ax1.set_axisbelow(True)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MSCE_C, label='MSCE'),
                   Patch(facecolor=GPT_C, label='GPT-5.5')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=11,
           framealpha=0.9, facecolor=BG, edgecolor='#1f2937', labelcolor=WHITE,
           handlelength=1.5).get_frame().set_linewidth(0.5)

ax1.set_title('各领域准确率对比', fontsize=18, fontweight='bold', color=WHITE, pad=12)
for spine in ax1.spines.values():
    spine.set_visible(False)

# ===== SECTION 2: Hallucination & Confidence (bottom 35%) =====
# 2a — Overall accuracy
ax2 = fig.add_axes([0.08, 0.04, 0.40, 0.27])
ax2.set_facecolor(CARD_BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Cards
ax2.text(5, 9.2, '综合准确率', fontsize=14, fontweight='bold', color=WHITE, ha='center')

# MSCE big number
ax2.text(2.2, 6.5, '91.7%', fontsize=36, fontweight='bold', color=MSCE_C, ha='center')
ax2.text(2.2, 5.2, 'MSCE', fontsize=12, color=SUB, ha='center')

# GPT big number
ax2.text(7.4, 6.5, '80.6%', fontsize=32, fontweight='bold', color=MUTED, ha='center')
ax2.text(7.4, 5.2, 'GPT-5.5', fontsize=12, color=MUTED, ha='center')

# vs line
ax2.plot([4.2, 5.8], [6.5, 6.5], color='#1f2937', linewidth=1, zorder=0)
ax2.text(5, 7.4, '386题', fontsize=9, color=MUTED, ha='center')

# Underline
ax2.axhline(y=3.5, xmin=0.15, xmax=0.85, color='#1f2937', linewidth=0.5)

# Sub-stat
ax2.text(1.8, 2.2, '前沿基准 100%', fontsize=11, color=GREEN, ha='center')
ax2.text(4.9, 2.2, '扩展基准 95%', fontsize=11, color=GREEN, ha='center')
ax2.text(8.1, 2.2, '206题 87.4%', fontsize=11, color=SUB, ha='center')

# 2b — Hallucination
ax3 = fig.add_axes([0.52, 0.04, 0.40, 0.27])
ax3.set_facecolor(CARD_BG)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

ax3.text(5, 9.2, '高置信度错误（幻觉）', fontsize=14, fontweight='bold', color=WHITE, ha='center')

# MSCE — 0
ax3.text(2.2, 6.5, '0', fontsize=52, fontweight='bold', color=GREEN, ha='center')
ax3.text(2.2, 5.2, 'MSCE', fontsize=12, color=SUB, ha='center')

# GPT — 54
ax3.text(7.4, 6.5, '54', fontsize=44, fontweight='bold', color=RED, ha='center')
ax3.text(7.4, 5.2, 'GPT-5.5', fontsize=12, color=MUTED, ha='center')

# vs
ax3.plot([4.2, 5.8], [6.5, 6.5], color='#1f2937', linewidth=1)

ax3.axhline(y=3.5, xmin=0.15, xmax=0.85, color='#1f2937', linewidth=0.5)

# Sub explanation
ax3.text(1.8, 2.5, '知道不确定', fontsize=10, color=GREEN, ha='center')
ax3.text(1.8, 1.5, '平均置信度 0.51', fontsize=9, color=MUTED, ha='center')

ax3.text(4.9, 2.5, '过度自信', fontsize=10, color=RED, ha='center')
ax3.text(4.9, 1.5, '平均置信度 0.74', fontsize=9, color=MUTED, ha='center')

ax3.text(8.1, 2.5, '跨386题', fontsize=10, color=SUB, ha='center')
ax3.text(8.1, 1.5, '3个基准套件', fontsize=9, color=MUTED, ha='center')

# ===== TOP TITLE =====
fig.text(0.5, 0.975, 'MSCE vs GPT-5.5', fontsize=22, fontweight='bold', color=WHITE, ha='center')
fig.text(0.5, 0.948, '6 模型对抗交叉验证 · 3 基准 · 386 题 · 6 领域', fontsize=11, color=SUB, ha='center')

# ===== FOOTER =====
fig.text(0.5, 0.005, 'MSCE 不生成内容。它在任何人之前告诉你内容在哪里断裂。',
         fontsize=9, color=MUTED, ha='center')

plt.savefig('/Users/dengxinhang/paper/constraint_residual/msce/assets/msce_benchmark_compare.png',
            dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
print("Done")
