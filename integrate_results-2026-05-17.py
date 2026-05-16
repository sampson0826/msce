#!/usr/bin/env python3
"""Integration script: parse downstream v3 + K=5 extension results and generate
paper-ready text snippets. Run after both experiments complete."""
import json, os, sys, math

BASE = os.path.dirname(os.path.abspath(__file__))

def mean(vals):
    return sum(vals)/len(vals) if vals else 0.0

def spearman_rho(xs, ys):
    def rank(vals):
        sp = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0]*len(vals)
        i = 0
        while i < len(sp):
            j = i
            while j < len(sp) and sp[j][1] == sp[i][1]:
                j += 1
            avg = (i+j-1)/2.0 + 1
            for k in range(i,j):
                ranks[sp[k][0]] = avg
            i = j
        return ranks
    if len(xs) < 3: return 0.0
    rx, ry = rank(xs), rank(ys)
    return pearson_r(rx, ry)

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx==0 or vy==0: return 0.0
    return cov/math.sqrt(vx*vy)


# ─── Downstream Code v3 ───────────────────────────────────────────
def process_downstream():
    path = os.path.join(BASE, "experiment_data", "downstream_code_v3_results.json")
    if not os.path.exists(path):
        print("[downstream] Results not found, skipping")
        return None

    with open(path) as f:
        data = json.load(f)

    corr = data.get("phase4_results", {}).get("correlation", {})
    per_model = data.get("phase4_results", {}).get("per_model", {})

    print("\n=== Downstream Code v3 Results ===")
    print(f"Models: {list(per_model.keys())}")
    print(f"Pearson r = {corr.get('pearson_r', 'N/A')}")
    print(f"Spearman rho = {corr.get('spearman_rho', 'N/A')}")
    print(f"p-value = {corr.get('p_value', 'N/A')}")

    for mname, info in sorted(per_model.items()):
        beta = info.get("beta", "N/A")
        qloss = info.get("quality_loss_per_gen", "N/A")
        beta_r2 = info.get("beta_r2", "N/A")
        print(f"  {mname:25s}  beta={str(beta):>10s}  q_loss/gen={str(qloss):>10s}  R2={str(beta_r2):>6s}")

    # Generate paper text snippet
    r = corr.get("pearson_r", 0)
    rho = corr.get("spearman_rho", 0)
    p = corr.get("p_value", 1.0)
    n_models = len(per_model)

    snippet = f"""
**Downstream code validation (v3).** {n_models} models evaluated on 35 Python coding
problems across 4 recursive generations. Constraint beta computed from pure text
features (HybridConstraintExtractor, no LLM judge). Code quality graded by
GPT-4o-mini (1-10 scale) for all 700 generated solutions. Correlation between
constraint beta and per-generation quality degradation: Pearson r = {r:.3f}
(p = {p:.3f}), Spearman rho = {rho:.3f}.
"""
    print("\n--- Paper Snippet ---")
    print(snippet)

    return {"corr": corr, "per_model": per_model, "snippet": snippet}


# ─── K=5 Extension ────────────────────────────────────────────────
def process_k5():
    k5dir = os.path.join(BASE, "experiment_data", "k5")
    ext_path = os.path.join(k5dir, "k5_extension_summary.json")
    analysis_path = os.path.join(k5dir, "k5_analysis.json")

    if not os.path.exists(ext_path):
        print("[k5] Extension summary not found, skipping")
        return None

    with open(ext_path) as f:
        ext = json.load(f)

    print("\n=== K=5 Extension Results ===")
    for name, info in ext.items():
        print(f"  {name}: beta={info.get('global_beta', 'N/A'):.6f}")

    # If analysis was re-run, read it
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
        results = analysis.get("results", {})

        # Count exponential wins for new models
        for name in ["gpt-4o-mini_k5", "claude-opus-4-7_k5"]:
            if name in results:
                r = results[name]
                exp_wins = sum(1 for cap, v in r.items()
                              if v.get("better_model") == "exponential")
                total = len([cap for cap in r if r[cap].get("better_model") != "insufficient_data"])
                print(f"  {name}: exponential wins {exp_wins}/{total}")

    # Generate paper snippet
    snippet = """
**K=5 extension beyond DeepSeek.** The K=5 depth experiment (§3.5) was extended from
3 DeepSeek-family models to include GPT-4o-mini (mid-range beta, 0.088) and Claude Opus 4.7
(high-beta, 0.164). Both models used the same 100-seed DeepSeek-V3 prompt set and the
same exponential-vs-linear comparison protocol. [Results to be filled in after analysis.]
"""
    print("\n--- Paper Snippet ---")
    print(snippet)

    return {"extension": ext, "snippet": snippet}


# ─── Main ─────────────────────────────────────────────────────────
def main():
    downstream = process_downstream()
    k5 = process_k5()

    # Combined summary path
    summary = {
        "downstream": downstream["corr"] if downstream else None,
        "k5_extension": k5["extension"] if k5 else None,
    }
    out = os.path.join(BASE, "experiment_data", "integration_summary-2026-05-17.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nIntegration summary saved to: {out}")


if __name__ == "__main__":
    main()
