#!/usr/bin/env python3
"""Full perplexity baseline — all 16 models, n=10 seeds each."""
import json, os, sys, time, math, re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "synthetic_decay_monitor"))

env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from synthetic_decay_monitor.provider_adapter import ProviderConfig, OpenAICompatibleAdapter

N_SEEDS, N_TOKENS, DELAY = 10, 15, 1.0
OUTPUT = os.path.join(BASE_DIR, "experiment_data", "perplexity_baseline_full.json")

MODELS = [
    ("gpt-5.5", "latest_models/gpt-5.5_s100_lineage.jsonl", "gpt-5.5_s100"),
    ("gemini-2.5-flash", "latest_models/gemini-2.5-flash_s100_lineage.jsonl", "gemini-2.5-flash_s100"),
    ("deepseek-v3", "n100/deepseek-chat_s100_lineage.jsonl", "deepseek-chat_s100"),
    ("deepseek-v4-flash", "latest_models/deepseek-v4-flash_s100_lineage.jsonl", "deepseek-v4-flash_s100"),
    ("llama-4-maverick", "latest_models/llama-4-maverick_s100_lineage.jsonl", "llama-4-maverick_s100"),
    ("deepseek-v4-pro", "latest_models/deepseek-v4-pro_s100_lineage.jsonl", "deepseek-v4-pro_s100"),
    ("llama-4-scout", "latest_models/llama-4-scout_s100_lineage.jsonl", "llama-4-scout_s100"),
    ("gpt-4o-mini", "n100/gpt-4o-mini_s100_lineage.jsonl", "gpt-4o-mini_s100"),
    ("llama-3.1-70b", "n100/meta-llama_llama-3_1-70b-instruct_s100_lineage.jsonl", "meta-llama_llama-3_1-70b-instruct_s100"),
    ("llama-3.1-8b", "n100/meta-llama_llama-3_1-8b-instruct_s100_lineage.jsonl", "meta-llama_llama-3_1-8b-instruct_s100"),
    ("gpt-4o", "n100/gpt-4o_s100_lineage.jsonl", "gpt-4o_s100"),
    ("deepseek-reasoner", "n100/deepseek-reasoner_s100_lineage.jsonl", "deepseek-reasoner_s100"),
    ("claude-sonnet-4-6", "n100/claude-sonnet-4-6_s100_lineage.jsonl", "claude-sonnet-4-6_s100"),
    ("claude-opus-4-6", "n100/claude-opus-4-6_s100_lineage.jsonl", "claude-opus-4-6_s100"),
    ("claude-haiku-4-5", "n100/claude-haiku-4-5-20251001_s100_lineage.jsonl", "claude-haiku-4-5-20251001_s100"),
    ("claude-opus-4-7", "latest_models/claude-opus-4-7_s100_lineage.jsonl", "claude-opus-4-7_s100"),
]

def mean(vals): return sum(vals)/len(vals) if vals else 0.0

def fit_beta(vals_by_gen):
    gens = sorted(vals_by_gen.keys())
    ys = [vals_by_gen[g] for g in gens]
    if len(gens) < 3: return 0.001, 0.0
    mv = mean(ys)
    if mv < 1e-6: return 0.001, 0.0
    stdv = (sum((y-mv)**2 for y in ys)/len(ys))**0.5
    if stdv/mv < 0.005: return 0.001, 0.0
    log_ys = [math.log(max(y, 1e-10)) for y in ys]
    mx = mean(gens)
    my = mean(log_ys)
    num = sum((g-mx)*(ly-my) for g,ly in zip(gens,log_ys))
    den = sum((g-mx)**2 for g in gens)
    if den == 0: return 0.001, 0.0
    slope = num/den
    beta = max(0.001, -slope)
    intercept = my - slope*mx
    ss_res = sum((ly-(intercept+slope*g))**2 for g,ly in zip(gens,log_ys))
    ss_tot = sum((ly-my)**2 for ly in log_ys)
    r2 = max(0.0, 1.0-ss_res/ss_tot) if ss_tot > 0 else 0.0
    return round(beta, 6), round(r2, 4)

def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    def rank(vals):
        sp = sorted((v,i) for i,v in enumerate(vals))
        ranks = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j < n and sp[j][0] == sp[i][0]: j += 1
            avg = 1.0 + (i+j-1)/2.0
            for k in range(i,j): ranks[sp[k][1]] = avg
            i = j
        return ranks
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1.0 - (6.0*d2)/(n*(n*n-1.0))

def sample_entries(entries, max_seeds):
    """Sample first max_seeds seeds, return all generations for those seeds."""
    seed_ids = []
    seen = set()
    for e in entries:
        raw = e.get("id", e.get("sample_id", ""))
        m = re.match(r'(?:G\d_)?(.+)', raw)
        sid = m.group(1) if m else raw
        if sid not in seen:
            seed_ids.append(sid)
            seen.add(sid)
    target = set(seed_ids[:max_seeds])
    result = []
    for e in entries:
        raw = e.get("id", e.get("sample_id", ""))
        m = re.match(r'(?:G\d_)?(.+)', raw)
        sid = m.group(1) if m else raw
        if sid in target:
            result.append(e)
    return result

def main():
    with open(os.path.join(BASE_DIR, "experiment_data/latest_models/all_models_summary.json")) as f:
        summary = json.load(f)
    beta_lookup = {k: v["global_beta"] for k, v in summary.items()}

    api_key = os.environ.get("QUICKROUTER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    config = ProviderConfig(base_url="https://api.quickrouter.ai/v1", api_key=api_key,
                            model="gpt-4o-mini", max_tokens=64, temperature=0.0, timeout_sec=90)
    adapter = OpenAICompatibleAdapter(config)
    print(f"Scoring via {adapter.provider_name}/{adapter.model}")

    all_results = {}
    for label, relpath, summary_key in MODELS:
        filepath = os.path.join(BASE_DIR, "experiment_data", relpath)
        if not os.path.exists(filepath):
            print(f"  [{label}] SKIP: file not found")
            continue

        print(f"\n── {label} ──")
        entries = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: entries.append(json.loads(line))
                except: continue

        sampled = sample_entries(entries, N_SEEDS)
        print(f"  {len(sampled)} entries from {N_SEEDS} seeds")
        
        gen_ppls = defaultdict(list)
        n_scored = 0
        for e in sampled:
            text = e.get("text", "")
            if not text or len(text) < 20: continue
            gen = e.get("generation", 0)
            result = adapter.continuation_perplexity(text, n_tokens=N_TOKENS)
            if result:
                gen_ppls[gen].append(result["perplexity"])
                n_scored += 1
            if n_scored % 20 == 0:
                print(f"  [{n_scored}/{len(sampled)}]")
            time.sleep(DELAY)

        gen_means = {g: mean(ppls) for g, ppls in gen_ppls.items()}
        beta_p, r2 = fit_beta(gen_means)
        beta_c = beta_lookup.get(summary_key)

        print(f"  G means: {' | '.join(f'G{g}: {gen_means[g]:.2f}' for g in sorted(gen_means))}")
        print(f"  β_perp={beta_p:.4f} R²={r2:.3f}  β_c={beta_c:.4f}" if beta_c else f"  β_perp={beta_p:.4f}")

        all_results[label] = {
            "constraint_beta": beta_c, "perplexity_beta": beta_p,
            "perplexity_r2": r2,
            "gen_mean_perplexity": {str(g): round(v,2) for g,v in gen_means.items()},
            "n_scored": n_scored,
        }

    models_both = [(k, v) for k, v in all_results.items() if v["constraint_beta"] is not None]
    c_betas = [v["constraint_beta"] for _, v in models_both]
    p_betas = [v["perplexity_beta"] for _, v in models_both]

    print(f"\n{'='*60}")
    print(f"  β_constraint vs β_perplexity (n={len(c_betas)})")
    for name, v in models_both:
        print(f"  {name:<22s} β_c={v['constraint_beta']:.4f}  β_perp={v['perplexity_beta']:.4f}")
    if len(c_betas) >= 3:
        rho = spearman_rho(c_betas, p_betas)
        print(f"\n  Spearman ρ = {rho:.4f}")

    with open(OUTPUT, "w") as f:
        json.dump({"models": all_results, "spearman_rho": rho if len(c_betas) >= 3 else None,
                   "n_models": len(c_betas)}, f, indent=2)
    print(f"\nSaved: {OUTPUT}")

if __name__ == "__main__":
    main()
