"""
LLM Judge 后端验证脚本 — 量化评测 embedding vs LLM judge 的 β 恢复精度。

用法:
    python eval_judge_backend.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --device cuda \\
        --output eval_results.json

2小时 GPU 窗口预算:
    - 模型加载: ~5-10 min
    - 样本评分: ~3-5 min (150 样本 × ~1-2s)
    - 约束提取 + 衰减分析: <1 min
    - 对比报告: <1 min
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from synthetic_decay_monitor.data_lineage import (
    generate_synthetic_lineage, parse_lineage_from_jsonl, DatasetLineage,
)
from synthetic_decay_monitor.constraint_extractor import (
    EmbeddingConstraintExtractor, LLMJudgeConstraintExtractor,
    HybridConstraintExtractor, extract_text_features, text_features_to_constraint,
)
from synthetic_decay_monitor.decay_engine import (
    DecayEngine, CapabilityStability, BASE_ALPHAS, S_CRITICAL,
)
from synthetic_decay_monitor.executor_classifier import (
    ExecutorClassifier, diagnose_executor_decay, CAPABILITY_EXECUTOR_PRIOR,
)


def create_judge_fn(model_name: str, device: str = "cuda"):
    """轻量级 LLM judge — 只做文本生成，不提取内部状态。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Judge] Loading {model_name} on {device} (fp16, sdpa)...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    print(f"[Judge] Loaded in {load_time:.0f}s, GPU mem: {gpu_mem:.1f} GB")

    def judge_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated.strip()

    return judge_fn


def run_backend_comparison(
    lineage: DatasetLineage,
    judge_fn=None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """在同一份数据上运行两种后端，输出对比结果。"""
    results = {
        "embedding": None,
        "llm_judge": None,
        "comparison": {},
    }

    # ---- Embedding 后端 ----
    print("\n" + "=" * 50)
    print("[1/2] Running Embedding backend...")
    t0 = time.time()
    try:
        emb_extractor = EmbeddingConstraintExtractor(embedding_model=embedding_model)
        gen0_texts = [s.text for s in lineage.generations.get(0, [])[:20]]
        if gen0_texts:
            emb_extractor.calibrate_fact_centroid(gen0_texts)
        emb_engine = DecayEngine(lineage, emb_extractor)
        emb_engine.run_all_capabilities()
        emb_time = time.time() - t0

        emb_trajectories = emb_engine.get_all_trajectories()
        emb_collapse = emb_engine.get_collapse_order()

        # 收集 embedding 的诊断
        emb_diagnoses = {}
        classifier = ExecutorClassifier()
        for cap, snapshots in emb_engine._snapshots.items():
            diag = diagnose_executor_decay(
                emb_engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            emb_diagnoses[cap] = diag

        results["embedding"] = {
            "time_seconds": emb_time,
            "trajectories": emb_trajectories,
            "collapse_order": emb_collapse,
            "diagnoses": {
                cap: {
                    "degradation_type": d["diagnosis"].degradation_type,
                    "severity": d["diagnosis"].severity,
                    "intervention": d["diagnosis"].intervention_type,
                    "beta": d["current_beta"],
                }
                for cap, d in emb_diagnoses.items()
            },
        }
        print(f"  Done in {emb_time:.0f}s")
    except Exception as e:
        print(f"  Embedding backend skipped: {e}")

    # ---- LLM Judge 后端 ----
    if judge_fn is not None:
        print("\n[2/2] Running LLM Judge backend...")
        t0 = time.time()
        judge_extractor = LLMJudgeConstraintExtractor(judge_fn=judge_fn)
        judge_engine = DecayEngine(lineage, judge_extractor)
        judge_engine.run_all_capabilities()
        judge_time = time.time() - t0

        judge_trajectories = judge_engine.get_all_trajectories()
        judge_collapse = judge_engine.get_collapse_order()

        judge_diagnoses = {}
        for cap, snapshots in judge_engine._snapshots.items():
            diag = diagnose_executor_decay(
                judge_engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            judge_diagnoses[cap] = diag

        results["llm_judge"] = {
            "time_seconds": judge_time,
            "trajectories": judge_trajectories,
            "collapse_order": judge_collapse,
            "diagnoses": {
                cap: {
                    "degradation_type": d["diagnosis"].degradation_type,
                    "severity": d["diagnosis"].severity,
                    "intervention": d["diagnosis"].intervention_type,
                    "beta": d["current_beta"],
                }
                for cap, d in judge_diagnoses.items()
            },
        }
        print(f"  Done in {judge_time:.0f}s")

    # ---- 对比分析 ----
    if results["embedding"] and results["llm_judge"]:
        comparison = _compute_comparison(results, lineage)
        results["comparison"] = comparison
        _print_comparison(comparison)

    return results


def _compute_comparison(results: dict, lineage: DatasetLineage) -> dict:
    emb = results["embedding"]
    judge = results["llm_judge"]

    # 收集每个 capability 的 β 值
    emb_betas = {}
    judge_betas = {}
    for t in emb["trajectories"]:
        if "trajectory" in t and t["trajectory"]:
            emb_betas[t["capability"]] = t["trajectory"][-1]["beta"]
    for t in judge["trajectories"]:
        if "trajectory" in t and t["trajectory"]:
            judge_betas[t["capability"]] = t["trajectory"][-1]["beta"]

    # β 跨度（高值-低值）= 区分度
    emb_spread = max(emb_betas.values()) - min(emb_betas.values()) if emb_betas else 0
    judge_spread = max(judge_betas.values()) - min(judge_betas.values()) if judge_betas else 0

    # β 排序与期望排序的相关性
    # 期望: math > code > logic > general/factual
    expected_order = ["math_reasoning", "code_generation", "logical_consistency"]
    emb_rank_ok = _check_rank_order(emb_betas, expected_order)
    judge_rank_ok = _check_rank_order(judge_betas, expected_order)

    # 诊断多样性（不同退化类型的数量）
    emb_diag_types = set(d["degradation_type"] for d in emb["diagnoses"].values())
    judge_diag_types = set(d["degradation_type"] for d in judge["diagnoses"].values())

    # 执行者构成分布
    judge_compositions = {}
    for t in judge["trajectories"]:
        if "trajectory" in t and t["trajectory"]:
            judge_compositions[t["capability"]] = t["trajectory"][-1].get("executor_composition", {})

    return {
        "beta_comparison": {
            cap: {"embedding": emb_betas.get(cap, 0), "llm_judge": judge_betas.get(cap, 0)}
            for cap in set(list(emb_betas.keys()) + list(judge_betas.keys()))
        },
        "beta_spread": {"embedding": emb_spread, "llm_judge": judge_spread},
        "spread_improvement": (judge_spread / emb_spread - 1) if emb_spread > 0 else 0,
        "rank_preservation": {"embedding": emb_rank_ok, "llm_judge": judge_rank_ok},
        "diagnosis_diversity": {
            "embedding": list(emb_diag_types),
            "llm_judge": list(judge_diag_types),
            "embedding_count": len(emb_diag_types),
            "llm_judge_count": len(judge_diag_types),
        },
        "judge_executor_compositions": judge_compositions,
        "time_ratio": results["llm_judge"]["time_seconds"] / max(results["embedding"]["time_seconds"], 1),
    }


def _check_rank_order(betas: dict, expected_high: list[str]) -> bool:
    """检查高 β 能力是否排在前列（宽松：只需前 2 高中有 2 个来自期望列表）。"""
    if len(betas) < 3:
        return True
    sorted_caps = sorted(betas, key=betas.get, reverse=True)
    top2 = set(sorted_caps[:2])
    hits = sum(1 for c in expected_high if c in top2)
    return hits >= 1


def _print_comparison(comparison: dict):
    print("\n" + "=" * 60)
    print("COMPARISON: Embedding vs LLM Judge")
    print("=" * 60)

    comp = comparison["beta_comparison"]
    print(f"\n{'Capability':<24} {'Embed β':>8} {'Judge β':>8} {'Δ':>8}")
    print("-" * 50)
    for cap, betas in sorted(comp.items()):
        delta = betas["llm_judge"] - betas["embedding"]
        print(f"{cap:<24} {betas['embedding']:8.3f} {betas['llm_judge']:8.3f} {delta:+8.3f}")

    print(f"\nβ Spread:")
    print(f"  Embedding: {comparison['beta_spread']['embedding']:.4f}")
    print(f"  LLM Judge: {comparison['beta_spread']['llm_judge']:.4f}")
    print(f"  Improvement: {comparison['spread_improvement']:+.0%}")

    print(f"\nDiagnosis Diversity:")
    print(f"  Embedding: {comparison['diagnosis_diversity']['embedding']} "
          f"({comparison['diagnosis_diversity']['embedding_count']} types)")
    print(f"  LLM Judge: {comparison['diagnosis_diversity']['llm_judge']} "
          f"({comparison['diagnosis_diversity']['llm_judge_count']} types)")

    print(f"\nSpeed: LLM Judge is {comparison['time_ratio']:.1f}x slower than Embedding")


# ================================================================
# β 恢复精度测试（金标准验证）
# ================================================================

def run_beta_recovery_test(
    judge_fn,
    target_betas: list[float] = None,
    n_samples: int = 30,
    n_generations: int = 4,
) -> dict:
    """注入已知 β 值，测试 LLM judge 的 β 恢复精度。

    对每个 target_beta 生成一组单一能力维度的合成数据，
    运行 decay engine 恢复 β，计算误差。
    """
    if target_betas is None:
        target_betas = [0.08, 0.15, 0.25, 0.35, 0.45]

    print("\n" + "=" * 60)
    print("BETA RECOVERY ACCURACY TEST")
    print("=" * 60)

    test_texts = [
        "The derivative of a function f at point x is defined as the limit of the difference quotient as h approaches zero. Therefore, we apply the chain rule to compute derivatives of composite functions because this forms the basis for gradient-based optimization in machine learning.",
        "Python is a high-level programming language supporting multiple paradigms including object-oriented and functional programming. However, dynamic typing can lead to runtime errors that would be caught at compile time in statically-typed languages such as Java or C++.",
        "The capital of France is Paris, with a population of approximately 2.1 million people within the city limits. The Eiffel Tower was completed in 1889 and stands 330 meters tall, attracting over 7 million visitors annually to the Île-de-France region.",
        "Therefore, based on the evidence presented across multiple independent studies, we can conclude that the hypothesis is strongly supported by empirical data. However, further research is needed to validate the findings across different demographic groups and geographic regions.",
        "In the quiet village nestled between rolling hills, the baker rose before dawn each morning, kneading dough with weathered hands that knew every curve and fold. His shop, established in 1957, served three generations of families with recipes passed down through centuries.",
    ] * (max(n_samples // 5, 1))

    results = {}
    judge_extractor = LLMJudgeConstraintExtractor(judge_fn=judge_fn)

    for beta_true in target_betas:
        print(f"\n  Testing β={beta_true:.2f}...")
        t0 = time.time()

        lineage = generate_synthetic_lineage(
            test_texts[:n_samples],
            n_generations=n_generations,
            decay_pattern={"*": beta_true},
        )
        engine = DecayEngine(lineage, judge_extractor)
        engine.run_all_capabilities()

        # 收集所有 capability 的恢复 β
        recovered_betas = {}
        for cap, traj in engine._trajectories.items():
            if traj:
                recovered_betas[cap] = traj[-1].beta

        avg_beta = np.mean(list(recovered_betas.values())) if recovered_betas else 0
        elapsed = time.time() - t0

        results[beta_true] = {
            "recovered_avg": avg_beta,
            "error": avg_beta - beta_true,
            "error_pct": (avg_beta - beta_true) / beta_true * 100 if beta_true > 0 else 0,
            "recovered_per_capability": recovered_betas,
            "time_seconds": elapsed,
        }
        print(f"    Recovered β={avg_beta:.4f} (error={avg_beta - beta_true:+.4f}, "
              f"{(avg_beta - beta_true)/beta_true*100:+.1f}%) [{elapsed:.0f}s]")

    # 汇总
    errors = [r["error"] for r in results.values()]
    abs_errors = [abs(e) for e in errors]
    pct_errors = [abs(r["error_pct"]) for r in results.values()]

    summary = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean([e**2 for e in errors]))),
        "mape": float(np.mean(pct_errors)),
        "max_error": float(max(abs_errors)),
        "correlation": float(np.corrcoef(
            list(results.keys()),
            [r["recovered_avg"] for r in results.values()]
        )[0, 1]) if len(results) >= 2 else 0,
    }

    print(f"\n  Summary: MAE={summary['mae']:.4f}, MAPE={summary['mape']:.1f}%, "
          f"Corr={summary['correlation']:.3f}")

    return {"per_beta": results, "summary": summary}


# ================================================================
# 执行者构成恢复测试（核心假设验证）
# ================================================================

# 期望映射：执行者构成 → 预期退化类型
_EXECUTOR_DIAG_MAP = {
    "pure_E-I": "E-I_loss",
    "pure_E-II": "E-II_loss",
    "pure_E-III": "E-III_loss",
    "balanced": "mixed",
}

# 纯退化模式测试：只施加一种执行者退化，验证 judge 能否识别单一退化类型
_EXECUTOR_TEST_CASES = {
    "pure_E-I": {"E-I": 1.0, "E-II": 0.0, "E-III": 0.0},
    "pure_E-II": {"E-I": 0.0, "E-II": 1.0, "E-III": 0.0},
    "pure_E-III": {"E-I": 0.0, "E-II": 0.0, "E-III": 1.0},
    "balanced": {"E-I": 0.33, "E-II": 0.33, "E-III": 0.34},
}


def run_executor_recovery_test(
    judge_fn,
    n_samples: int = 30,
    n_generations: int = 4,
    beta: float = 0.25,
) -> dict:
    """注入已知执行者构成，验证 LLM judge 能否恢复正确的退化类型。

    对每个测试用例生成退化数据（保持 β 恒定，仅改变执行者构成），
    检查诊断结果是否匹配预期退化类型。
    """
    print("\n" + "=" * 60)
    print("EXECUTOR COMPOSITION RECOVERY TEST")
    print("=" * 60)
    print(f"  β fixed at {beta}, testing {len(_EXECUTOR_TEST_CASES)} executor mixes")

    test_texts = [
        "The derivative of a function f at point x is defined as the limit of the difference quotient as h approaches zero. Therefore, we apply the chain rule to compute derivatives of composite functions because this forms the basis for gradient-based optimization in machine learning.",
        "Python is a high-level programming language supporting multiple paradigms including object-oriented and functional programming. However, dynamic typing can lead to runtime errors that would be caught at compile time in statically-typed languages such as Java or C++.",
        "The capital of France is Paris, with a population of approximately 2.1 million people within the city limits. The Eiffel Tower was completed in 1889 and stands 330 meters tall, attracting over 7 million visitors annually to the Île-de-France region.",
        "Therefore, based on the evidence presented across multiple independent studies, we can conclude that the hypothesis is strongly supported by empirical data. However, further research is needed to validate the findings across different demographic groups and geographic regions.",
        "In the quiet village nestled between rolling hills, the baker rose before dawn each morning, kneading dough with weathered hands that knew every curve and fold. His shop, established in 1957, served three generations of families with recipes passed down through centuries.",
    ] * (max(n_samples // 5, 1))

    results = {}
    judge_extractor = LLMJudgeConstraintExtractor(judge_fn=judge_fn)
    classifier = ExecutorClassifier()

    for case_name, exec_mix in _EXECUTOR_TEST_CASES.items():
        expected_diag = _EXECUTOR_DIAG_MAP[case_name]
        print(f"\n  Testing {case_name}: {exec_mix} (expect: {expected_diag})")
        t0 = time.time()

        lineage = generate_synthetic_lineage(
            test_texts[:n_samples],
            n_generations=n_generations,
            decay_pattern={"*": beta},
            executor_pattern={"*": exec_mix},
        )
        engine = DecayEngine(lineage, judge_extractor)
        engine.run_all_capabilities()

        # 收集诊断结果
        diagnoses = {}
        recovered_compositions = {}
        for cap, snapshots in engine._snapshots.items():
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            diagnoses[cap] = {
                "degradation_type": diag["diagnosis"].degradation_type,
                "severity": diag["diagnosis"].severity,
                "intervention": diag["diagnosis"].intervention_type,
                "helmholtz_scalar": diag["diagnosis"].helmholtz_scalar_potential,
                "helmholtz_vector": diag["diagnosis"].helmholtz_vector_potential,
            }
            if engine._trajectories.get(cap):
                recovered_compositions[cap] = engine._trajectories[cap][-1].executor_composition

        # 汇总诊断
        diag_types = [d["degradation_type"] for d in diagnoses.values()]
        # 多数投票
        from collections import Counter
        majority_diag = Counter(diag_types).most_common(1)[0][0]
        match = majority_diag == expected_diag

        elapsed = time.time() - t0

        results[case_name] = {
            "injected_composition": exec_mix,
            "expected_diagnosis": expected_diag,
            "actual_diagnoses": diagnoses,
            "majority_diagnosis": majority_diag,
            "match": match,
            "recovered_compositions": recovered_compositions,
            "time_seconds": elapsed,
        }
        status = "MATCH" if match else f"MISMATCH (got {majority_diag})"
        print(f"    {status} | diagnoses: {diag_types} [{elapsed:.0f}s]")

    # 汇总统计
    n_correct = sum(1 for r in results.values() if r["match"])
    accuracy = n_correct / len(results) if results else 0

    # 计算各执行者维度的恢复相关性
    comp_corrs = {}
    for exec_type in ["E-I", "E-II", "E-III"]:
        injected = []
        recovered = []
        for case_name, r in results.items():
            injected.append(_EXECUTOR_TEST_CASES[case_name][exec_type])
            avg_rec = np.mean([
                comp.get(exec_type, 0.33)
                for comp in r["recovered_compositions"].values()
            ]) if r["recovered_compositions"] else 0.33
            recovered.append(avg_rec)
        if len(injected) >= 2:
            comp_corrs[exec_type] = float(np.corrcoef(injected, recovered)[0, 1])
        else:
            comp_corrs[exec_type] = 0

    summary = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(results),
        "composition_correlation": comp_corrs,
    }

    print(f"\n  Accuracy: {n_correct}/{len(results)} = {accuracy:.1%}")
    print(f"  Composition correlation: E-I={comp_corrs['E-I']:.3f}, "
          f"E-II={comp_corrs['E-II']:.3f}, E-III={comp_corrs['E-III']:.3f}")

    return {"per_case": results, "summary": summary}


def run_executor_recovery_test_hybrid(
    judge_fn=None,
    n_samples: int = 30,
    n_generations: int = 4,
    beta: float = 0.25,
) -> dict:
    """使用 HybridConstraintExtractor 的执行者构成恢复测试。

    Hybrid = 文本特征（E-I/E-II）+ 可选 LLM judge（E-III）。
    可在纯文本模式（judge_fn=None）下运行，无需 GPU。
    """
    print("\n" + "=" * 60)
    print("HYBRID EXECUTOR COMPOSITION RECOVERY TEST")
    print("=" * 60)
    mode = "text_features + LLM judge" if judge_fn else "text_features only"
    print(f"  Mode: {mode} | β={beta} | {len(_EXECUTOR_TEST_CASES)} cases")

    # 事实型文本：同时包含逻辑连接词（E-I）、多样化词汇（E-II）、专名和数字（E-III）
    test_texts = [
        "The capital of France is Paris, which has a population of approximately 2.1 million people within the city limits. Therefore, the metropolitan area faces significant housing challenges. However, the government has invested 500 million euros in transportation infrastructure because commute times have increased since 2010.",
        "The Eiffel Tower was completed in 1889 and stands 330 meters tall, attracting over 7 million visitors annually to the Île-de-France region. Consequently, tourism generates billions of euros in revenue for the local economy. Nevertheless, the COVID-19 pandemic reduced visitor numbers to just 1.5 million in 2020.",
        "Albert Einstein published his theory of special relativity in 1905 while working at the Swiss Patent Office in Bern. His famous equation E=mc² established the equivalence of mass and energy. Furthermore, his general theory of relativity was completed in 1915 and confirmed during the solar eclipse of 1919.",
        "The Amazon rainforest covers approximately 5.5 million square kilometers across nine South American countries. Brazil contains about 60 percent of the forest within its borders. However, deforestation has claimed over 17 percent of the original forest area since 1970 according to satellite data from Brazil's National Institute for Space Research.",
        "The Great Wall of China extends over 21,000 kilometers across northern China and was constructed over multiple dynasties spanning 2,000 years. The Ming Dynasty section near Beijing receives approximately 10 million visitors each year. However, remote sections in Gansu province see fewer than 1,000 visitors annually due to their inaccessibility.",
    ] * (max(n_samples // 5, 1))

    results = {}
    hybrid_extractor = HybridConstraintExtractor(judge_fn=judge_fn)
    classifier = ExecutorClassifier()

    for case_name, exec_mix in _EXECUTOR_TEST_CASES.items():
        expected_diag = _EXECUTOR_DIAG_MAP[case_name]
        print(f"\n  Testing {case_name}: {exec_mix} (expect: {expected_diag})")
        t0 = time.time()

        lineage = generate_synthetic_lineage(
            test_texts[:n_samples],
            n_generations=n_generations,
            decay_pattern={"*": beta},
            executor_pattern={"*": exec_mix},
            capability_tags=["factual_knowledge"],
        )
        engine = DecayEngine(lineage, hybrid_extractor)
        engine.run_all_capabilities()

        diagnoses = {}
        recovered_compositions = {}
        for cap, snapshots in engine._snapshots.items():
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            diagnoses[cap] = {
                "degradation_type": diag["diagnosis"].degradation_type,
                "severity": diag["diagnosis"].severity,
                "intervention": diag["diagnosis"].intervention_type,
            }
            if engine._trajectories.get(cap):
                recovered_compositions[cap] = engine._trajectories[cap][-1].executor_composition

        from collections import Counter
        diag_types = [d["degradation_type"] for d in diagnoses.values()]
        majority_diag = Counter(diag_types).most_common(1)[0][0]
        match = majority_diag == expected_diag

        elapsed = time.time() - t0

        results[case_name] = {
            "injected_composition": exec_mix,
            "expected_diagnosis": expected_diag,
            "actual_diagnoses": diagnoses,
            "majority_diagnosis": majority_diag,
            "match": match,
            "recovered_compositions": recovered_compositions,
            "time_seconds": elapsed,
        }
        status = "MATCH" if match else f"MISMATCH (got {majority_diag})"
        print(f"    {status} | diagnoses: {diag_types} [{elapsed:.0f}s]")

    n_correct = sum(1 for r in results.values() if r["match"])
    accuracy = n_correct / len(results) if results else 0

    comp_corrs = {}
    for exec_type in ["E-I", "E-II", "E-III"]:
        injected = []
        recovered = []
        for case_name, r in results.items():
            injected.append(_EXECUTOR_TEST_CASES[case_name][exec_type])
            avg_rec = np.mean([
                comp.get(exec_type, 0.33)
                for comp in r["recovered_compositions"].values()
            ]) if r["recovered_compositions"] else 0.33
            recovered.append(avg_rec)
        if len(injected) >= 2:
            comp_corrs[exec_type] = float(np.corrcoef(injected, recovered)[0, 1])
        else:
            comp_corrs[exec_type] = 0

    summary = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(results),
        "composition_correlation": comp_corrs,
        "mode": mode,
    }

    print(f"\n  [Hybrid] Accuracy: {n_correct}/{len(results)} = {accuracy:.1%}")
    print(f"  [Hybrid] Composition correlation: E-I={comp_corrs['E-I']:.3f}, "
          f"E-II={comp_corrs['E-II']:.3f}, E-III={comp_corrs['E-III']:.3f}")

    return {"per_case": results, "summary": summary}


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Judge Backend Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--device", default="cuda",
                       help="Device: cuda, cpu, mps")
    parser.add_argument("--output", default="eval_results.json",
                       help="Output JSON path")
    parser.add_argument("--demo-n", type=int, default=6,
                       help="Demo generations for full pipeline test")
    parser.add_argument("--beta-test", action="store_true", default=True,
                       help="Run β recovery accuracy test")
    parser.add_argument("--no-beta-test", action="store_false", dest="beta_test",
                       help="Skip β recovery test (faster)")
    parser.add_argument("--executor-test", action="store_true", default=True,
                       help="Run executor composition recovery test")
    parser.add_argument("--no-executor-test", action="store_false", dest="executor_test",
                       help="Skip executor recovery test (faster)")
    parser.add_argument("--extractor", default="llm",
                       choices=["llm", "embedding", "hybrid", "hybrid+llm", "all"],
                       help="Extractor backend: llm, embedding, hybrid (text features), "
                            "hybrid+llm (text + LLM), all (compare)")
    parser.add_argument("--input", default=None,
                       help="Real JSONL data (optional)")
    args = parser.parse_args()

    # ---- 加载 Judge 模型（按需） ----
    print("=" * 60)
    print("LLM Judge Backend Evaluation")
    print(f"Extractor: {args.extractor}")
    print("=" * 60)

    judge_fn = None
    if args.extractor in ("llm", "hybrid+llm", "all"):
        print(f"Loading LLM: {args.model} on {args.device}")
        judge_fn = create_judge_fn(args.model, args.device)

    all_results = {
        "model": args.model,
        "device": args.device,
        "extractor": args.extractor,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- 1. β 恢复精度测试 (仅 LLM judge) ----
    if args.beta_test and judge_fn is not None:
        beta_results = run_beta_recovery_test(judge_fn)
        all_results["beta_recovery"] = beta_results

    # ---- 2. 执行者构成恢复测试 ----
    if args.executor_test:
        if args.extractor in ("llm", "all") and judge_fn is not None:
            executor_results = run_executor_recovery_test(judge_fn)
            all_results["executor_recovery_llm"] = executor_results

        if args.extractor in ("hybrid", "hybrid+llm", "all"):
            hybrid_judge = judge_fn if args.extractor == "hybrid+llm" else None
            hybrid_results = run_executor_recovery_test_hybrid(
                judge_fn=hybrid_judge,
                n_samples=5,       # 快速模式：5 样本
                n_generations=3,   # 3 代
            )
            key = "executor_recovery_hybrid_llm" if hybrid_judge else "executor_recovery_hybrid"
            all_results[key] = hybrid_results

    # ---- 3. 全管线对比 ----
    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPARISON")
    print(f"  Extractor: {args.extractor}")
    print("=" * 60)

    if args.input:
        lineage = parse_lineage_from_jsonl(args.input)
    else:
        demo_texts = [
            "The derivative of x^2 is 2x. To solve the equation, we apply the chain rule.",
            "Python functions are defined using the def keyword. They can accept arguments and return values.",
            "The capital of France is Paris. The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
            "Therefore, based on the evidence presented, we can conclude that the hypothesis is supported.",
            "In the quiet village, the baker rose before dawn each day, kneading dough with hands that knew every curve.",
            "The water cycle involves evaporation, condensation, and precipitation. These processes are driven by solar energy.",
            "To optimize the algorithm, we can use dynamic programming to cache intermediate results and reduce complexity.",
            "The novel explores themes of identity and belonging through the eyes of a narrator who has lost both.",
            "Please provide a step-by-step solution showing all work and explaining the reasoning at each step.",
            "The study followed 10,000 participants over 20 years, tracking cardiovascular outcomes against dietary patterns.",
        ] * 3
        lineage = generate_synthetic_lineage(demo_texts, n_generations=args.demo_n)

    # Run selected backend(s)
    if args.extractor == "hybrid":
        print("\n[Hybrid] Running text-feature-only pipeline...")
        t0 = time.time()
        hybrid_extractor = HybridConstraintExtractor(judge_fn=None)
        hybrid_engine = DecayEngine(lineage, hybrid_extractor)
        hybrid_engine.run_all_capabilities()
        hybrid_time = time.time() - t0
        all_results["hybrid_pipeline"] = {
            "time_seconds": hybrid_time,
            "trajectories": hybrid_engine.get_all_trajectories(),
            "collapse_order": hybrid_engine.get_collapse_order(),
        }
        print(f"  Done in {hybrid_time:.0f}s")
    elif args.extractor == "hybrid+llm":
        print("\n[Hybrid+LLM] Running text-features + LLM judge pipeline...")
        t0 = time.time()
        hybrid_extractor = HybridConstraintExtractor(judge_fn=judge_fn)
        hybrid_engine = DecayEngine(lineage, hybrid_extractor)
        hybrid_engine.run_all_capabilities()
        hybrid_time = time.time() - t0
        all_results["hybrid_llm_pipeline"] = {
            "time_seconds": hybrid_time,
            "trajectories": hybrid_engine.get_all_trajectories(),
            "collapse_order": hybrid_engine.get_collapse_order(),
        }
        print(f"  Done in {hybrid_time:.0f}s")
    elif args.extractor == "embedding":
        print("\n[Embedding] Running embedding pipeline...")
        t0 = time.time()
        emb_extractor = EmbeddingConstraintExtractor()
        gen0_texts = [s.text for s in lineage.generations.get(0, [])[:20]]
        if gen0_texts:
            emb_extractor.calibrate_fact_centroid(gen0_texts)
        emb_engine = DecayEngine(lineage, emb_extractor)
        emb_engine.run_all_capabilities()
        emb_time = time.time() - t0
        all_results["embedding_pipeline"] = {
            "time_seconds": emb_time,
            "trajectories": emb_engine.get_all_trajectories(),
            "collapse_order": emb_engine.get_collapse_order(),
        }
        print(f"  Done in {emb_time:.0f}s")
    elif args.extractor == "all":
        comparison = run_backend_comparison(lineage, judge_fn)
        all_results["pipeline_comparison"] = comparison
        # Also run hybrid
        hybrid_results = run_executor_recovery_test_hybrid(
            judge_fn=judge_fn, n_samples=5, n_generations=3,
        )
        all_results["executor_recovery_hybrid_llm"] = hybrid_results
    else:
        comparison = run_backend_comparison(lineage, judge_fn)
        all_results["pipeline_comparison"] = comparison

    # ---- 保存 ----
    # 清理 non-serializable 对象
    output = _make_serializable(all_results)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    return 0


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    sys.exit(main())
