# Constraint Residual Framework

**The DESI DR2 CMB+BAO Evidence for Dynamical Dark Energy is a Projection Artifact of the CMB Geometric Degeneracy**

Deng Xinhang (Independent Research)

## Paper

- **arXiv**: [pending]
- **PDF**: `paper_desi_artifact/main.pdf`
- **中文版**: `paper_desi_artifact/main_cn.pdf`

## Abstract

The DESI DR2 CMB+BAO $2.3\sigma$ preference for $w_0 w_a$CDM is a projection artifact. Constraint residual decomposition reveals the tension is strictly 1D (PC1 > 99.99%), aligned with the CMB geometric degeneracy. A single shift along this direction within $\Lambda$CDM yields $\Delta\chi^2 = 22.16$, more than twice the $w_0 w_a$ extension ($\Delta\chi^2 = 9.7$, Wilks-corrected to $\sim 2.6\sigma$).

## Repository Structure

```
constraint_residual/
├── paper_desi_artifact/          # LaTeX source + figures + compiled PDF
│   ├── main.tex                  # English paper
│   ├── main_cn.tex               # Chinese version
│   ├── fig1_chisq_scan.pdf       # Figure 1: chi-squared scan
│   └── fig2_eigenvalue_spectrum.pdf  # Figure 2: eigenvalue spectrum
├── experiment_data/              # Pre-computed results (JSON)
│   ├── verify_desi_tension_20260522_153952.json  # chi-squared scan data
│   ├── three_proofs_20260522_155322.json         # Proof A/B/C results
│   ├── desi_planck_tension_20260522_145629.json  # B-operator + tension directions
│   └── figures/                  # PNG versions of figures
├── css_system/
│   └── make_figures.py           # Figure generation script
└── README.md
```

## Reproducing the Results

### Verify numbers from JSON

All quantitative claims in the paper can be independently verified by reading the JSON files in `experiment_data/`. See the [Layer 0 provenance report](experiment_data/) for the complete field-level mapping of paper claims → JSON data.

### Regenerate figures

```bash
pip install matplotlib numpy
python3 css_system/make_figures.py
```

Output: `experiment_data/figures/fig1_chisq_scan.{pdf,png}` and `fig2_eigenvalue_spectrum.{pdf,png}`

### Reproduce from scratch

The full analysis pipeline consists of:

1. **Planck posterior**: Download Planck 2018 baseline $\Lambda$CDM MCMC chains from [PLA](https://pla.esac.esa.int) (plikHM_TTTEEE_lowl_lowE_lensing)
2. **$r_d$ calibration**: Analytic approximation calibrated against CLASS (see Eq. 4 + coefficients in paper)
3. **DESI BAO likelihood**: Implement DESI DR2 BAO distance likelihood using data from [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
4. **Importance sampling**: Reweight Planck chain samples by $\exp(-\chi^2_{\mathrm{DESI}}/2)$
5. **B-operator**: Compute $\mathcal{B} = \Sigma_P^{-1/2}(\Sigma_P - \Sigma_{PD})\Sigma_P^{-1/2}$ and its eigendecomposition
6. **$\chi^2$ scan**: Evaluate $\chi^2_{\mathrm{DESI}}$ along PC1 direction

The pipeline source code is available at `css_system/`.

## Data Sources

| Source | URL | Reference |
|--------|-----|-----------|
| Planck 2018 | [PLA](https://pla.esac.esa.int) | Planck Collaboration (2020), A&A 641, A6 |
| DESI DR2 BAO | [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data) | DESI Collaboration (2025), arXiv:2503.14738 |
| CLASS | [github.com/lesgourg/class_public](https://github.com/lesgourg/class_public) | Blas et al. (2011), Lesgourgues (2011) |

## Citation

If you use these results, please cite:

```bibtex
@article{deng2026desi,
  title={The DESI DR2 CMB+BAO Evidence for Dynamical Dark Energy is a Projection Artifact of the CMB Geometric Degeneracy},
  author={Deng, Xinhang},
  year={2026},
  note={arXiv:XXXX.XXXXX}
}
```

## License

MIT
