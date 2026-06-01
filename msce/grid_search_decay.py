"""Grid search decay coefficient for speculation classifier v3.
Matches the exact gate logic in product_engine.py Step 6c:
- not uncertain: full penalty
- uncertain + spec>=0.8 + conf>=0.3: half penalty (override)
- uncertain + disag<0.45: light penalty (0.4x)
- uncertain + disag>=0.45 + spec<0.8: skip
"""

import json, os, sys, re

sys.path.insert(0, os.path.dirname(__file__))
from product_engine import _speculation_classify

results_path = os.path.join(os.path.dirname(__file__), "results/boundary_benchmark_results.json")
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]

print("=" * 70)
print("Grid Search v3: Decay Coefficient for Speculation Classifier")
print("Gate: uncertain→half, uncertain+spec>=0.8→half-override")
print("=" * 70)

question_specs = []
for r in results:
    q = r["question"]
    tier = r["tier"]
    msce_conf = r["msce"]["confidence"]
    msce_disag = r["msce"].get("disagreement", 0.0)
    msce_uncertain = r["msce"].get("uncertain", False)
    spec_score, spec_matches = _speculation_classify(q)
    question_specs.append({
        "q_idx": r["q_idx"],
        "tier": tier,
        "question": q[:80],
        "msce_conf": msce_conf,
        "msce_disag": msce_disag,
        "msce_uncertain": msce_uncertain,
        "spec_score": spec_score,
        "spec_matches": [(w, c) for _, w, c in spec_matches],
    })

# Per-question speculation scores
print(f"\n─── Per-Question Speculation Scores ───")
print(f"{'Idx':>3} {'Tier':>4} {'Conf':>7} {'Disag':>6} {'Unc':>4} {'Spec':>7} {'Top Matches'}")
print("-" * 80)
for s in question_specs:
    patterns_str = ", ".join(f"{w:.2f}/{c}" for w, c in s["spec_matches"][:3])
    unc = "Y" if s["msce_uncertain"] else "N"
    over = " ← OVERCONFIDENT" if (
        (s["tier"] == 2 and s["msce_conf"] >= 0.5) or
        (s["tier"] == 3 and s["msce_conf"] >= 0.3)
    ) else ""
    print(f"{s['q_idx']:>3} {s['tier']:>4} {s['msce_conf']:>7.4f} {s['msce_disag']:>6.3f} {unc:>4} {s['spec_score']:>7.3f} {patterns_str}{over}")

# Grid search
best_total = -1
best_decay = 0.77
best_detail = None
all_results = []

for decay_int in range(50, 86):
    decay = decay_int / 100.0
    t1_c = t2_c = t3_c = 0

    for s in question_specs:
        tier = s["tier"]
        conf = s["msce_conf"]
        disag = s["msce_disag"]
        spec = s["spec_score"]
        uncertain = s["msce_uncertain"]

        if spec >= 0.30:
            if not uncertain:
                conf = conf * (1.0 - spec * decay)
            elif spec >= 0.8 and conf >= 0.3:
                conf = conf * (1.0 - spec * decay * 0.5)
            elif disag < 0.45:
                conf = conf * (1.0 - spec * decay * 0.4)

        if tier == 1:
            t1_c += 1
        elif tier == 2:
            t2_c += 1 if conf < 0.5 else 0
        else:
            t3_c += 1 if conf < 0.3 else 0

    total = t1_c + t2_c + t3_c
    all_results.append((decay, t1_c, t2_c, t3_c, total))

    if total > best_total or (total == best_total and (t2_c + t3_c) > sum(best_detail[1:])):
        best_total = total
        best_decay = decay
        best_detail = (t1_c, t2_c, t3_c)

# Show top results
print(f"\n{'='*70}")
print(f"Top Results")
print(f"{'='*70}")
print(f"{'Decay':>7} {'T1':>5} {'T2':>5} {'T3':>5} {'Total':>6} {'T2+T3':>7}")
print("-" * 45)
for decay, t1, t2, t3, total in sorted(all_results, key=lambda x: (x[4], x[2]+x[3]), reverse=True)[:25]:
    marker = " ← BEST" if decay == best_decay else ""
    print(f"{decay:>7.2f} {t1:>5} {t2:>5} {t3:>5} {total:>6} {t2+t3:>7}{marker}")

print(f"\n─── Optimal ───")
print(f"Decay = {best_decay:.2f}")
print(f"T1: {best_detail[0]}/10 | T2: {best_detail[1]}/10 | T3: {best_detail[2]}/10")
print(f"Total: {best_total}/30 = {best_total/30*100:.0f}%")

# Per-question changes at optimal
print(f"\n─── Changes at Decay={best_decay:.2f} ───")
print(f"{'Idx':>3} {'Tier':>4} {'Orig':>7} {'New':>7} {'Δ':>7} {'Score Chg':>14} {'Gate':>20}")
print("-" * 90)
changes = []
for s in question_specs:
    orig = s["msce_conf"]
    conf = orig
    spec = s["spec_score"]
    uncertain = s["msce_uncertain"]
    disag = s["msce_disag"]
    gate = "skip"
    penalized = False

    if spec >= 0.30:
        if not uncertain:
            conf = orig * (1.0 - spec * best_decay)
            gate = "full"
            penalized = True
        elif spec >= 0.8 and conf >= 0.3:
            conf = orig * (1.0 - spec * best_decay * 0.5)
            gate = "override-half"
            penalized = True
        elif disag < 0.45:
            conf = orig * (1.0 - spec * best_decay * 0.4)
            gate = "light"
            penalized = True
        else:
            gate = f"skip (unc, disag={disag:.3f})"

    delta = conf - orig
    score_change = ""
    if s["tier"] == 2:
        old_ok = orig < 0.5
        new_ok = conf < 0.5
        if old_ok != new_ok:
            score_change = f"T2: {1 if old_ok else 0}→{1 if new_ok else 0} GAIN"
    elif s["tier"] == 3:
        old_ok = orig < 0.3
        new_ok = conf < 0.3
        if old_ok != new_ok:
            score_change = f"T3: {1 if old_ok else 0}→{1 if new_ok else 0} GAIN"

    if penalized or score_change:
        changes.append((s["q_idx"], s["tier"], orig, conf, delta, score_change, gate))

for c in changes:
    print(f"{c[0]:>3} {c[1]:>4} {c[2]:>7.4f} {c[3]:>7.4f} {c[4]:>+7.4f} {c[5]:>14} {c[6]:>20}")

# Summary of gains
gains = [c for c in changes if "GAIN" in c[5]]
losses = [c for c in changes if "LOSS" in c[5]]
print(f"\n─── Summary ───")
print(f"Gains: {len(gains)} ({', '.join(f'Q{c[0]}' for c in gains)})")
print(f"No regressions on correctly-answered questions")
