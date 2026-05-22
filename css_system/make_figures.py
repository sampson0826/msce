#!/usr/bin/env python3
"""
Generate publication-quality figures for the Constraint Residual framework.
Figure 1: Chi-squared scan along geometric degeneracy direction.
Figure 2: Eigenvalue spectrum of Delta_Sigma (DESI - Planck).
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global matplotlib rcParams -- clean academic style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VERIFY_PATH = (
    "/Users/dengxinhang/paper/constraint_residual/experiment_data/"
    "verify_desi_tension_20260522_153952.json"
)
EIGEN_PATH = (
    "/Users/dengxinhang/paper/constraint_residual/experiment_data/"
    "desi_planck_tension_20260522_145629.json"
)
OUT_DIR = "/Users/dengxinhang/paper/constraint_residual/experiment_data/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(VERIFY_PATH) as fh:
    verify = json.load(fh)

with open(EIGEN_PATH) as fh:
    eigen = json.load(fh)

# ===================================================================
# FIGURE 1: Chi-squared scan along geometric degeneracy direction
# ===================================================================
scan = verify['scan']
lambdas = np.array([s['lambda'] for s in scan])
chisq_desi = np.array([s['chisq_desi'] for s in scan])
chisq_planck = np.array([s['chisq_planck'] for s in scan])

# Colors
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GRAY = '#666666'
C_BLACK = '#111111'

fig1, ax1 = plt.subplots(figsize=(8, 5))

# -- Left Y-axis: chi2_DESI ------------------------------------------------
line_desi, = ax1.plot(lambdas, chisq_desi, '-', color=C_BLUE, linewidth=2.0,
                      label=r'$\chi^2_{\rm DESI}$')
ax1.set_xlabel(r'$\lambda$ (movement along geometric degeneracy, $\sigma$ units)')
ax1.set_ylabel(r'$\chi^2_{\rm DESI}$', color=C_BLUE, fontsize=14)
ax1.tick_params(axis='y', labelcolor=C_BLUE)
ax1.set_xlim(-3.0, 3.0)

# -- Right Y-axis: chi2_Planck ---------------------------------------------
ax2 = ax1.twinx()
line_pl, = ax2.plot(lambdas, chisq_planck, '--', color=C_RED, linewidth=2.0,
                    label=r'$\chi^2_{\rm Planck}$')
ax2.set_ylabel(r'$\chi^2_{\rm Planck}$', color=C_RED, fontsize=14)
ax2.tick_params(axis='y', labelcolor=C_RED)

# -- Horizontal line: 2-sigma threshold for 13 dof -------------------------
chi2_thresh = 22.36
ax1.axhline(y=chi2_thresh, color=C_GRAY, linestyle='--', linewidth=1.0, alpha=0.75)
ax1.text(-2.85, chi2_thresh + 0.6, r'$\chi^2_{\rm thresh}=22.36$ (2$\sigma$, 13 d.o.f.)',
         fontsize=10, color=C_GRAY, va='bottom')

# -- Vertical line: Planck best-fit (lambda = 0) ---------------------------
ax1.axvline(x=0, color=C_BLACK, linestyle=':', linewidth=1.0, alpha=0.55)
ax1.text(0.08, 92, r'Planck best-fit  ($\lambda = 0$)',
         fontsize=10, color=C_BLACK, va='top', ha='left')

# -- Vertical line + annotation: lambda = +1.0 (chi2 drops below 2-sigma) --
idx_l1 = np.argmin(np.abs(lambdas - 1.0))
chi2_at_l1 = chisq_desi[idx_l1]
ax1.axvline(x=1.0, color=C_BLUE, linestyle=':', linewidth=1.0, alpha=0.55)
ax1.annotate(
    r'$\lambda = +1.0$' + '\n' + r'$\chi^2_{\rm DESI} = %.1f < 22.36$' % chi2_at_l1,
    xy=(1.0, chi2_at_l1),
    xytext=(1.95, 38),
    fontsize=10,
    color=C_BLUE,
    arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.3),
    ha='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_BLUE, alpha=0.7),
)

# -- Combined legend -------------------------------------------------------
lines = [line_desi, line_pl]
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc='upper right')

# -- Title and final -------------------------------------------------------
ax1.set_title(r'$\chi^2$ Scan along Geometric Degeneracy Direction',
              fontsize=14, pad=10)

fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_chisq_scan.pdf'), format='pdf')
fig1.savefig(os.path.join(OUT_DIR, 'fig1_chisq_scan.png'), format='png')
plt.close(fig1)

# ===================================================================
# FIGURE 2: Eigenvalue spectrum of Delta_Sigma
# ===================================================================
td = eigen['results']['tension_directions']

# Build bar data from the three principal components
pc_labels = ['PC1', 'PC2', 'PC3']
pc_values = [
    td['pc1_variance_pct'],
    td['pc2_variance_pct'],
    td['pc3_variance_pct'],
]

# Colors for the bars
bar_colors = ['#2166AC', '#D1E5F0', '#D1E5F0']

fig2, ax = plt.subplots(figsize=(7, 4))

x_pos = np.arange(len(pc_labels))
bars = ax.bar(x_pos, pc_values, color=bar_colors, edgecolor='black',
              linewidth=0.9, width=0.55)

# -- Value labels on bars --------------------------------------------------
for bar, val in zip(bars, pc_values):
    h = bar.get_height()
    if val > 1.0:
        ax.text(bar.get_x() + bar.get_width() / 2., h + 1.5,
                '%.1f%%' % val,
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3,
                '~0%%' if val < 0.01 else '%.1f%%' % val,
                ha='center', va='bottom', fontsize=11, color=C_GRAY)

# -- "1D Tension" annotation -----------------------------------------------
ax.annotate(
    '1D Tension',
    xy=(0, pc_values[0]),
    xytext=(1.0, pc_values[0] * 0.65),
    fontsize=14,
    fontweight='bold',
    color=C_BLUE,
    arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=2.2),
    ha='center',
)

# -- Axes decoration -------------------------------------------------------
ax.set_xticks(x_pos)
ax.set_xticklabels(pc_labels)
ax.set_xlabel('Principal Component', fontsize=14)
ax.set_ylabel('Variance Explained (%)', fontsize=14)
ax.set_title(r'Eigenvalue Spectrum of $\Delta\Sigma$  (DESI $-$ Planck)',
             fontsize=14, pad=10)
ax.set_ylim(0, max(pc_values) * 1.22)

# Remove top + right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_eigenvalue_spectrum.pdf'), format='pdf')
fig2.savefig(os.path.join(OUT_DIR, 'fig2_eigenvalue_spectrum.png'), format='png')
plt.close(fig2)

# ===================================================================
print(f"Figures saved to {OUT_DIR}")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname}  ({size_kb:.1f} KB)")
