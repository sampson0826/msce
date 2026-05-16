#!/usr/bin/env python3
"""Fix perplexity beta: use |slope| (perplexity increase = degradation)."""
import json, math

def fix_fit_beta(vals_by_gen):
    """Re-fit beta using |slope| for perplexity growth."""
    gens = sorted(int(k) for k in vals_by_gen.keys())
    ys = [vals_by_gen[str(g)] for g in gens]
    if len(gens) < 3:
        return 0.001, 0.0
    mv = sum(ys) / len(ys)
    if mv < 1e-6:
        return 0.001, 0.0
    stdv = (sum((y - mv)**2 for y in ys) / len(ys))**0.5
    if stdv / mv < 0.005:
        return 0.001, 0.0
    
    log_ys = [math.log(max(y, 1e-10)) for y in ys]
    mx = sum(gens) / len(gens)
    my = sum(log_ys) / len(log_ys)
    num = sum((g - mx)*(ly - my) for g, ly in zip(gens, log_ys))
    den = sum((g - mx)**2 for g in gens)
    if den == 0:
        return 0.001, 0.0
    
    slope = num / den
    # Use |slope| because perplexity increase = more degradation
    beta = max(0.001, abs(slope))
    
    intercept = my - slope * mx
    ss_res = sum((ly - (intercept + slope*g))**2 for g, ly in zip(gens, log_ys))
    ss_tot = sum((ly - my)**2 for ly in log_ys)
    r2 = max(0.0, 1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    return round(beta, 6), round(r2, 4)

def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    def rank(vals):
        sp = sorted((v, i) for i, v in enumerate(vals))
        ranks = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j < n and sp[j][0] == sp[i][0]: j += 1
            avg = 1.0 + (i+j-1)/2.0
            for k in range(i, j): ranks[sp[k][1]] = avg
            i = j
        return ranks
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1.0 - (6.0*d2)/(n*(n*n-1.0))

# Load and fix
for fname in ['experiment_data/perplexity_baseline.json',
              'experiment_data/perplexity_baseline_full.json']:
    try:
        with open(fname) as f:
            data = json.load(f)
    except FileNotFoundError:
        continue
    
    models = data.get('models', {})
    fixed = {}
    for label, v in models.items():
        gen_means = v.get('gen_mean_perplexity', {})
        if len(gen_means) >= 3:
            beta_p, r2 = fix_fit_beta(gen_means)
            v['perplexity_beta'] = beta_p
            v['perplexity_r2'] = r2
            fixed[label] = beta_p
    
    # Recompute Spearman
    models_both = [(k, v) for k, v in models.items()
                   if v.get('constraint_beta') is not None and v.get('perplexity_beta', 0) > 0]
    c_betas = [v['constraint_beta'] for _, v in models_both]
    p_betas = [v['perplexity_beta'] for _, v in models_both]
    
    if len(c_betas) >= 3:
        rho = spearman_rho(c_betas, p_betas)
        data['spearman_rho'] = rho
        data['n_models_with_both'] = len(c_betas)
    
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'Fixed: {fname}')
    print(f'  n={len(fixed)} betas recomputed')
    for label, beta_p in sorted(fixed.items()):
        print(f'    {label}: β_perp={beta_p:.4f}')
    if len(c_betas) >= 3:
        print(f'  Spearman ρ = {rho:.4f}')
    print()
