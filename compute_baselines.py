"""Compute text-degradation baselines (entropy, TTR, repetition) on existing
lineage data and compare against β rankings.  Zero API cost — pure computation.
"""
import json, math
from collections import Counter
from pathlib import Path

LINEAGE_DIR = Path("experiment_data/latest_models")
N100_DIR = Path("experiment_data/n100")
SUMMARY_PATH = LINEAGE_DIR / "all_models_summary.json"

# Direct mapping: paper model name -> lineage filename stem (without _lineage)
# Files are in latest_models/ or n100/
MODEL_MAP = {
    "GPT-5.5":                   "gpt-5.5_s100",
    "Gemini 2.5 Flash":          "gemini-2.5-flash_s100",
    "DeepSeek-V3 (Chat)":        "deepseek-chat_s100",
    "DeepSeek-V4 Flash":         "deepseek-v4-flash_s100",
    "Llama 4 Maverick":          "llama-4-maverick_s100",
    "DeepSeek-V4 Pro":           "deepseek-v4-pro_s100",
    "Llama 4 Scout":             "llama-4-scout_s100",
    "GPT-4o-mini":               "gpt-4o-mini_s100",
    "Llama 3.1 70B":             "meta-llama_llama-3_1-70b-instruct_s100",
    "Llama 3.1 8B":              "meta-llama_llama-3_1-8b-instruct_s100",
    "GPT-4o":                    "gpt-4o_s100",
    "DeepSeek-R1":               "deepseek-reasoner_s100",
    "Claude Sonnet 4.6":         "claude-sonnet-4-6_s100",
    "Claude Opus 4.6":           "claude-opus-4-6_s100",
    "Claude Haiku 4.5":          "claude-haiku-4-5-20251001_s100",
    "Claude Opus 4.7":           "claude-opus-4-7_s100",
}

def find_file(stem: str) -> Path:
    for d in [LINEAGE_DIR, N100_DIR]:
        p = d / f"{stem}_lineage.jsonl"
        if p.exists():
            return p
    return None

def shannon_entropy(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 2:
        return 0.0
    counts = Counter(chars)
    total = len(chars)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def type_token_ratio(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 2:
        return 1.0
    return len(set(tokens)) / len(tokens)

def compute_metrics(jsonl_path: Path) -> dict:
    gen_texts = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            gen = rec.get("generation", 0)
            text = rec.get("text", "")
            if text and gen < 4:  # Gen 0,1,2,3
                gen_texts.setdefault(gen, []).append(text)
    results = {}
    for gen in sorted(gen_texts):
        texts = gen_texts[gen]
        ents = [shannon_entropy(t) for t in texts]
        ttrs = [type_token_ratio(t) for t in texts]
        results[gen] = {
            "entropy_mean": sum(ents)/len(ents),
            "ttr_mean": sum(ttrs)/len(ttrs),
            "n": len(texts),
        }
    return results

def fit_beta(values, gens):
    """Fit exponential C_k = C_0*(1-beta)^k. Returns floor if flat."""
    import numpy as np
    mean_v = sum(values)/len(values)
    if mean_v < 1e-6:
        return 0.001
    cv = (sum((v-mean_v)**2 for v in values)/len(values))**0.5 / mean_v
    if cv < 0.005:
        return 0.001
    ys = np.log(np.maximum(np.array(values), 1e-10))
    slope = np.polyfit(np.array(gens), ys, 1)[0]
    return max(0.001, min(1.0 - math.exp(slope), 0.999))

def spearman_rho(xs, ys):
    n = len(xs)
    def rank(vals):
        sp = sorted((v,i) for i,v in enumerate(vals))
        ranks = [0]*n
        i = 0
        while i < n:
            j = i
            while j < n and sp[j][0] == sp[i][0]: j += 1
            avg = 1 + (i+j-1)/2
            for k in range(i,j): ranks[sp[k][1]] = avg
            i = j
        return ranks
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1.0 - (6.0*d2)/(n*(n*n-1.0))

def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    # Build lookup: model name from filename base -> summary beta
    # summary keys: e.g. "deepseek-v4-pro_s100", "gpt-4o-mini_s100", etc.
    beta_lookup = {}
    for k, v in summary.items():
        beta_lookup[k] = v["global_beta"]

    results = []
    for name, stem in MODEL_MAP.items():
        fp = find_file(stem)
        if not fp:
            print(f"  MISSING: {name} ({stem})")
            continue
        metrics = compute_metrics(fp)
        gens = sorted(metrics.keys())
        ent_vals = [metrics[g]["entropy_mean"] for g in gens]
        ttr_vals = [metrics[g]["ttr_mean"] for g in gens]

        ent_beta = fit_beta(ent_vals, gens)
        ttr_beta = fit_beta(ttr_vals, gens)

        # Get constraint beta
        constraint_beta = None
        # Try exact match first
        if stem in beta_lookup:
            constraint_beta = beta_lookup[stem]
        else:
            # Try fuzzy
            for k, v in beta_lookup.items():
                if stem.replace("_s100","") in k or k.replace("_s100","") in stem:
                    constraint_beta = v
                    break

        results.append({
            "model": name,
            "constraint_beta": constraint_beta,
            "entropy_beta": round(ent_beta, 6),
            "ttr_beta": round(ttr_beta, 6),
            "ent_traj": [round(metrics[g]["entropy_mean"], 4) for g in gens],
            "ttr_traj": [round(metrics[g]["ttr_mean"], 4) for g in gens],
        })
        status = "✓" if constraint_beta else "?"
        print(f"  {status} {name}: β_c={constraint_beta}, β_ent={ent_beta:.4f}, β_ttr={ttr_beta:.4f}")

    # Spearman
    c_betas = [r["constraint_beta"] for r in results if r["constraint_beta"]]
    for metric in ["entropy_beta", "ttr_beta"]:
        m_betas = [r[metric] for r in results if r["constraint_beta"]]
        rho = spearman_rho(c_betas, m_betas)
        print(f"\nSpearman ρ(β_constraint, {metric}): {rho:.3f}")

    # Simple table for paper
    print("\n=== PAPER TABLE (sorted by constraint β) ===")
    print(f"{'Model':<28} {'β_c':>6} {'β_ent':>6} {'β_ttr':>6}")
    print("-"*50)
    for r in sorted(results, key=lambda x: x["constraint_beta"] or 0):
        print(f"{r['model']:<28} {r['constraint_beta']:>6.4f} {r['entropy_beta']:>6.4f} {r['ttr_beta']:>6.4f}")

    out = Path("experiment_data/baseline_comparison.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
