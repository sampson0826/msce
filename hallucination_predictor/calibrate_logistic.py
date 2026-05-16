"""
Calibrate logistic regression: Δ||Π|| → P(hallucination).

Reads poc_results_7b.json, fits logistic regression with 2-fold CV,
and prints the actual α, β parameters. Does NOT require GPU/model.

Usage:
  python -m constraint_residual.hallucination_predictor.calibrate_logistic
"""

import json, sys, os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def main():
    data_path = Path(__file__).parent / "output" / "poc_results_7b.json"
    with open(data_path) as f:
        data = json.load(f)

    results = data["results"]
    scores = np.array([r["residual_jump"] for r in results])
    labels = np.array([1 if r["is_hallucination"] else 0 for r in results])

    n = len(scores)
    n_hallu = labels.sum()
    print(f"n={n}, n_hallucination={n_hallu}, n_correct={n - n_hallu}")

    # ---- 2-fold cross-validation ----
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    all_alphas, all_betas = [], []
    all_aucs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(scores, labels)):
        X_train, X_test = scores[train_idx].reshape(-1, 1), scores[test_idx].reshape(-1, 1)
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, class_weight=None)
        clf.fit(X_train, y_train)

        alpha = float(clf.intercept_[0])
        beta = float(clf.coef_[0][0])
        all_alphas.append(alpha)
        all_betas.append(beta)

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
        all_aucs.append(auc)

        print(f"Fold {fold+1}: α={alpha:.4f}, β={beta:.4f}, test AUC={auc:.4f}")

    print(f"\nPooled: α={np.mean(all_alphas):.4f}, β={np.mean(all_betas):.4f}")
    print(f"Mean test AUC: {np.mean(all_aucs):.4f}")

    # ---- Also fit on ALL data for reference ----
    clf_all = LogisticRegression(max_iter=1000, class_weight=None)
    clf_all.fit(scores.reshape(-1, 1), labels)
    alpha_all = float(clf_all.intercept_[0])
    beta_all = float(clf_all.coef_[0][0])
    print(f"\nFull-data fit: α={alpha_all:.4f}, β={beta_all:.4f}")

    # ---- Score distribution check ----
    print(f"\nScore range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"Hallu scores:  mean={scores[labels==1].mean():.6f} std={scores[labels==1].std():.6f}")
    print(f"Correct scores: mean={scores[labels==0].mean():.6f} std={scores[labels==0].std():.6f}")

    # Test points
    print("\nCalibration check:")
    for s in [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16]:
        p = 1.0 / (1.0 + np.exp(-(alpha_all + beta_all * s)))
        print(f"  Δ||Π||={s:+.4f} → P(hallu)={p:.4f}")


if __name__ == "__main__":
    main()
