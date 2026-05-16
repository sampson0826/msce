#!/usr/bin/env python3
"""
Downstream Validation v2 — Pure LLM-Judge Extractor
====================================================

Replaces the hybrid extractor (run_downstream_validation.py) with a pure
LLM-judge constraint extractor (MultiJudgeExtractor from pure_judge_extractor.py).

Phase 1: Generate recursive lineage for 3 models x 3 task types x 10 problems
Phase 2: Extract constraint scores via MultiJudgeExtractor (gpt-4o-mini, 1 repeat)
Phase 3: Grade task quality via GPT-4o-mini LLM judge (1-10 scale)
Phase 4: Correlate constraint beta with quality degradation

Output: experiment_data/downstream_v2_results.json
"""

import json
import os
import re
import sys
import time
import math
import traceback
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "synthetic_decay_monitor"))

# Load .env first
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)

from synthetic_decay_monitor.provider_adapter import create_provider, ProviderConfig
from synthetic_decay_monitor.constraint_extractor import ConstraintState, _safe_float

# MultiJudgeExtractor from pure_judge_extractor
from pure_judge_extractor import (
    MultiJudgeExtractor, DIMENSIONS,
    parse_scores, scores_to_constraint_state,
    EXTRACTION_PROMPT, build_judge_fn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_DATA = os.path.join(BASE_DIR, "experiment_data")
CACHE_DIR = os.path.join(EXPERIMENT_DATA, "downstream_v2_cache")
OUTPUT_PATH = os.path.join(EXPERIMENT_DATA, "downstream_v2_results.json")
os.makedirs(CACHE_DIR, exist_ok=True)

GENERATIONS = 3  # Gen1, Gen2, Gen3

# Models under test and their provider configs
MODEL_CONFIGS = {
    "gpt-4o-mini": {"provider": "quickrouter", "model": "gpt-4o-mini"},
    "deepseek-v3": {"provider": "deepseek", "model": "deepseek-chat"},
    "claude-sonnet-4-6": {"provider": "quickrouter", "model": "claude-sonnet-4-6"},
}

TASK_TYPES = ["code_generation", "math_reasoning", "factual_knowledge"]

# ---------------------------------------------------------------------------
# New problem sets (distinct from existing seeds and downstream problems)
# ---------------------------------------------------------------------------

CODE_PROBLEMS = [
    ("count_words", "Write a Python function `count_words(s)` that returns the number of words in a string. Words are separated by spaces."),
    ("find_max", "Write a Python function `find_max(lst)` that returns the maximum element in a list without using the built-in max() function."),
    ("matrix_transpose", "Write a Python function `transpose(matrix)` that returns the transpose of a 2D matrix represented as a list of lists."),
    ("my_atoi", "Write a Python function `my_atoi(s)` that converts a string to an integer, handling leading whitespace and optional +/- sign. Do not use int()."),
    ("power_of_two", "Write a Python function `is_power_of_two(n)` that returns True if the integer n is a power of two."),
    ("caesar_cipher", "Write a Python function `caesar_cipher(text, shift)` that applies a Caesar cipher to the string text with the given shift (0-25)."),
    ("longest_word", "Write a Python function `longest_word(sentence)` that returns the longest word in a sentence. If there's a tie, return the first one."),
    ("sum_digits", "Write a Python function `sum_digits(n)` that returns the sum of the digits of a non-negative integer n."),
    ("group_anagrams", "Write a Python function `group_anagrams(words)` that groups a list of strings into lists of anagrams. Return a list of lists."),
    ("roman_to_int", "Write a Python function `roman_to_int(s)` that converts a Roman numeral string (I, V, X, L, C, D, M) to an integer."),
]

MATH_PROBLEMS = [
    ("compound_interest", "A $1000 investment earns 5% compound interest annually. How much is it worth after 3 years? Show your calculation step by step."),
    ("card_probability", "From a standard 52-card deck, what is the probability of drawing an ace or a heart? Show your work clearly."),
    ("solve_system", "Solve the system of equations: 3x + 2y = 12, 5x - 3y = 1. Show all steps."),
    ("geometric_series", "What is the sum of the geometric series: 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384? Show your calculation."),
    ("permutations_letters", "How many distinct ways can you arrange the letters in the word 'SCIENCE'? Show your calculation."),
    ("log_equation", "Solve for x: log_2(x) + log_2(x-2) = 3. Show all steps."),
    ("derivative_product", "Find the derivative of f(x) = x^2 * e^x using the product rule. Show each step."),
    ("bayes_disease", "A disease affects 1% of the population. A test is 95% accurate for detecting the disease (true positive rate) and has a 2% false positive rate. If someone tests positive, what is the probability they actually have the disease? Show Bayes' theorem calculation."),
    ("train_meeting", "A train leaves Station A heading east at 60 km/h. Another train leaves Station B (300 km east of A) heading west at 90 km/h at the same time. When and where do they meet? Show your work."),
    ("modular_exponent", "Compute 7^100 mod 13 using modular arithmetic properties. Show your reasoning step by step."),
]

FACT_PROBLEMS = [
    ("longest_coastline", "Which country has the longest coastline in the world? Provide a concise answer with brief context."),
    ("higgs_boson", "What is the Higgs boson and why is it significant in particle physics? Answer concisely in 2-3 sentences."),
    ("one_hundred_years", "Who wrote 'One Hundred Years of Solitude' and what literary style is it most associated with? Answer concisely."),
    ("krebs_cycle", "What is the Krebs cycle (citric acid cycle) and where in the cell does it occur? Answer concisely."),
    ("treaty_westphalia", "What was the Peace of Westphalia (1648) and why is it considered a foundational event in international relations? Answer concisely."),
    ("weather_vs_climate", "What is the difference between weather and climate? Explain in 2-3 sentences."),
    ("four_forces", "What are the four fundamental forces of nature, ordered from strongest to weakest?"),
    ("rosetta_stone", "What is the Rosetta Stone and why was it significant for understanding ancient Egypt? Answer concisely."),
    ("rna_vs_dna", "What are two key differences between RNA and DNA in terms of structure and function? Answer concisely."),
    ("meiji_restoration", "What was the Meiji Restoration (1868) and why was it transformative for Japan? Answer concisely."),
]

PROBLEM_SETS = {
    "code_generation": CODE_PROBLEMS,
    "math_reasoning": MATH_PROBLEMS,
    "factual_knowledge": FACT_PROBLEMS,
}

# ---------------------------------------------------------------------------
# Grading prompts (LLM-as-judge for downstream task quality)
# ---------------------------------------------------------------------------

GRADE_CODE_PROMPT = """You are a code quality evaluator. Score the following answer to a coding problem on a 1-10 scale.

Scoring criteria:
- 10: Perfect solution. Correct algorithm, well-structured, handles edge cases, clean code.
- 8-9: Correct solution with minor issues (style, minor edge case).
- 6-7: Mostly correct logic but has bugs or missing edge cases.
- 4-5: Partially correct approach but significant errors.
- 2-3: Some relevant code but fundamentally wrong approach.
- 1: Completely wrong or nonsensical code.

Problem: {problem}

Answer to evaluate:
---
{answer}
---

Reply with ONLY a single integer score (1-10):"""

GRADE_MATH_PROMPT = """You are a math reasoning evaluator. Score the following answer to a math problem on a 1-10 scale.

Scoring criteria:
- 10: Perfect solution with correct final answer and rigorous step-by-step reasoning.
- 8-9: Correct answer with mostly valid reasoning, minor gaps.
- 6-7: Right approach but calculation error or incomplete reasoning.
- 4-5: Partially correct approach, significant errors in reasoning.
- 2-3: Some relevant math concepts but fundamentally wrong.
- 1: Completely wrong answer with no valid reasoning.

Problem: {problem}

Answer to evaluate:
---
{answer}
---

Reply with ONLY a single integer score (1-10):"""

GRADE_FACT_PROMPT = """You are a factual accuracy evaluator. Score the following answer to a factual question on a 1-10 scale.

Scoring criteria:
- 10: All claims are factually accurate, precise, and well-articulated.
- 8-9: Mostly accurate with one minor imprecision.
- 6-7: Partially accurate but missing key details or has one significant error.
- 4-5: Mix of accurate and inaccurate claims.
- 2-3: Mostly inaccurate with only tangential relevance.
- 1: Completely wrong or unrelated claims.

Question: {problem}

Answer to evaluate:
---
{answer}
---

Reply with ONLY a single integer score (1-10):"""

GRADE_PROMPTS = {
    "code_generation": GRADE_CODE_PROMPT,
    "math_reasoning": GRADE_MATH_PROMPT,
    "factual_knowledge": GRADE_FACT_PROMPT,
}

# ---------------------------------------------------------------------------
# Helper: robust API call with retry and exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(adapter, prompt: str, max_tokens: int = 512,
                    temperature: float = 0.7, max_retries: int = 4,
                    base_delay: float = 2.0, label: str = "") -> str:
    """Call an LLM adapter with retry + exponential backoff.

    Handles: rate limits (429), timeouts, connection errors, empty responses.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = adapter.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if resp and len(resp.strip()) >= 1:
                return resp.strip()
            # Empty response — retry with longer delay
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    [{label}] empty response, retry {attempt+1}/{max_retries} in {delay:.0f}s...")
                time.sleep(delay)
        except Exception as e:
            last_err = e
            err_str = str(e)[:150]
            status_429 = "429" in err_str or "rate" in err_str.lower()
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if status_429:
                    delay = max(delay, 10.0)  # rate limits need longer wait
                print(f"    [{label}] {type(e).__name__}: {err_str} — retry {attempt+1}/{max_retries} in {delay:.0f}s")
                time.sleep(delay)
            else:
                print(f"    [{label}] FAILED after {max_retries} attempts: {type(e).__name__}: {err_str}")

    return "[ERROR]" if not last_err else f"[ERROR: {type(last_err).__name__}]"


# ---------------------------------------------------------------------------
# Phase 1: Generate recursive lineage
# ---------------------------------------------------------------------------

def generate_lineage(model_key: str, task_type: str, problems: list,
                     adapter, delay: float = 1.0) -> list[dict]:
    """Generate recursive lineage for one model on one task type.

    Returns list of dicts: {model, task_type, problem_id, problem_statement,
                            generation, text}
    """
    results = []
    n = len(problems)
    task_label = f"{model_key}/{task_type}"

    for pi, (prob_id, prob_stmt) in enumerate(problems):
        current_text = prob_stmt  # Gen0: problem statement
        for gen in range(1, GENERATIONS + 1):
            label = f"{task_label}/{prob_id}_gen{gen}"
            resp = call_with_retry(
                adapter, current_text,
                max_tokens=512, temperature=0.7, label=label,
            )
            results.append({
                "model": model_key,
                "task_type": task_type,
                "problem_id": prob_id,
                "problem_statement": prob_stmt,
                "generation": gen,
                "text": resp,
            })
            current_text = resp  # recursive: output → next input
            time.sleep(delay)

        if (pi + 1) % 3 == 0 or pi == n - 1:
            print(f"  [{task_label}] {pi+1}/{n} problems done")

    return results


def run_phase1(resume_from: Optional[list] = None) -> list[dict]:
    """Generate recursive lineage for all model x task combinations.

    Args:
        resume_from: Previously saved lineage entries (for resumption).

    Returns:
        Full list of lineage entries.
    """
    print("\n" + "=" * 70)
    print("  Phase 1: Generate Recursive Lineage")
    print("=" * 70)

    # Determine what's already done
    done_keys = set()
    if resume_from:
        for entry in resume_from:
            done_keys.add((entry["model"], entry["task_type"], entry["problem_id"],
                          entry["generation"]))

    all_entries = list(resume_from) if resume_from else []

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n  Model: {model_key} (via {cfg['provider']})")
        try:
            adapter = create_provider(cfg["provider"], model=cfg["model"])
        except Exception as e:
            print(f"  SKIP: Cannot create adapter for {model_key}: {e}")
            continue

        for task_type in TASK_TYPES:
            problems = PROBLEM_SETS[task_type]

            # Check if already fully done
            needed = sum(1 for p in problems
                        for g in range(1, GENERATIONS + 1)
                        if (model_key, task_type, p[0], g) not in done_keys)
            if needed == 0:
                print(f"  [{model_key}/{task_type}] already complete, skipping")
                continue

            print(f"  [{model_key}/{task_type}] {needed} generations needed "
                  f"({len(problems)} problems x {GENERATIONS} gens)")

            try:
                entries = generate_lineage(
                    model_key, task_type, problems, adapter, delay=1.0,
                )
                # Filter out already-done entries (shouldn't be any, but be safe)
                for e in entries:
                    k = (e["model"], e["task_type"], e["problem_id"], e["generation"])
                    if k not in done_keys:
                        all_entries.append(e)
                        done_keys.add(k)

                # Save checkpoint after each task type
                save_checkpoint("phase1_lineage.json", all_entries)
                print(f"  [{model_key}/{task_type}] done. Checkpoint saved.")

            except Exception as e:
                print(f"  ERROR [{model_key}/{task_type}]: {type(e).__name__}: {e}")
                traceback.print_exc()
                save_checkpoint("phase1_lineage.json", all_entries)
                print(f"  Checkpoint saved, continuing with next task.")

    print(f"\n  Phase 1 complete: {len(all_entries)} total lineage entries")
    return all_entries


# ---------------------------------------------------------------------------
# Phase 2: Extract constraint scores via MultiJudgeExtractor
# ---------------------------------------------------------------------------

def extract_constraints(lineage: list[dict],
                        resume_from: Optional[list] = None) -> list[dict]:
    """Extract constraint scores for all lineage entries.

    Uses MultiJudgeExtractor with gpt-4o-mini only, 1 repeat.

    Args:
        lineage: Phase 1 lineage entries.
        resume_from: Previously saved extraction entries.

    Returns:
        List of dicts with dimension_scores, constraint_state, total_constraint.
    """
    print("\n" + "=" * 70)
    print("  Phase 2: Extract Constraint Scores (Pure LLM-Judge)")
    print("=" * 70)

    # Build extractor with single judge
    print("  Building MultiJudgeExtractor (gpt-4o-mini, 1 repeat)...")
    extractor = MultiJudgeExtractor(
        judge_names=["gpt-4o-mini"],
        n_repeats=1,
    )

    if "gpt-4o-mini" not in extractor.judge_fns:
        print("  FATAL: gpt-4o-mini judge not available")
        return []

    # Determine what's already done
    done_keys = set()
    all_extractions = []
    if resume_from:
        for entry in resume_from:
            key = (entry["model"], entry["task_type"], entry["problem_id"],
                   entry["generation"])
            done_keys.add(key)
            all_extractions.append(entry)

    # Process entries
    to_process = [e for e in lineage
                  if (e["model"], e["task_type"], e["problem_id"], e["generation"])
                  not in done_keys]

    print(f"  {len(to_process)} entries to extract, {len(done_keys)} already cached")

    for i, entry in enumerate(to_process):
        text = entry["text"]
        text_id = f"{entry['model']}/{entry['task_type']}/{entry['problem_id']}_gen{entry['generation']}"

        try:
            result = extractor.extract_sample(
                text_id=text_id,
                text=text,
                capability=entry["task_type"],
                generation=entry["generation"],
            )

            # Serialize the result
            total = extractor.compute_total_constraint(result.constraint_state_mean)
            extraction_entry = {
                "model": entry["model"],
                "task_type": entry["task_type"],
                "problem_id": entry["problem_id"],
                "generation": entry["generation"],
                "dimension_scores": {
                    j: {d: round(s, 6) for d, s in scores.items()}
                    for j, scores in result.scores_by_judge.items()
                },
                "constraint_state": {
                    "sigma_fact": round(result.constraint_state_mean.sigma_fact, 6),
                    "sigma_syntax": round(result.constraint_state_mean.sigma_syntax, 6),
                    "sigma_style": round(result.constraint_state_mean.sigma_style, 6),
                    "sigma_safety": round(result.constraint_state_mean.sigma_safety, 6),
                    "sigma_coherence": round(result.constraint_state_mean.sigma_coherence, 6),
                },
                "total_constraint": round(total, 6),
            }
            all_extractions.append(extraction_entry)
            done_keys.add((entry["model"], entry["task_type"], entry["problem_id"],
                          entry["generation"]))

        except Exception as e:
            print(f"  [WARN] Extraction failed for {text_id}: {type(e).__name__}: {e}")
            # Use fallback values
            fallback_scores = {d: 0.5 for d in DIMENSIONS}
            fallback_state = scores_to_constraint_state(fallback_scores)
            total = extractor.compute_total_constraint(fallback_state)
            extraction_entry = {
                "model": entry["model"],
                "task_type": entry["task_type"],
                "problem_id": entry["problem_id"],
                "generation": entry["generation"],
                "dimension_scores": {"gpt-4o-mini": fallback_scores},
                "constraint_state": {
                    "sigma_fact": 0.5, "sigma_syntax": 0.5,
                    "sigma_style": 0.5, "sigma_safety": 0.5, "sigma_coherence": 0.5,
                },
                "total_constraint": round(total, 6),
            }
            all_extractions.append(extraction_entry)

        # Progress and checkpoint
        if (i + 1) % 15 == 0:
            print(f"  Processed {i+1}/{len(to_process)} entries...")
            save_checkpoint("phase2_extractions.json", all_extractions)

        time.sleep(0.5)  # rate limit safety

    # Final checkpoint
    save_checkpoint("phase2_extractions.json", all_extractions)
    print(f"  Phase 2 complete: {len(all_extractions)} extractions")
    return all_extractions


# ---------------------------------------------------------------------------
# Phase 3: Grade task quality via LLM judge
# ---------------------------------------------------------------------------

def grade_quality(lineage: list[dict],
                  resume_from: Optional[list] = None) -> list[dict]:
    """Grade answer quality for all lineage entries using GPT-4o-mini.

    Args:
        lineage: Phase 1 lineage entries.
        resume_from: Previously saved grading entries.

    Returns:
        List of dicts with quality_score.
    """
    print("\n" + "=" * 70)
    print("  Phase 3: Grade Task Quality (GPT-4o-mini Judge)")
    print("=" * 70)

    # Create grader adapter
    try:
        grader = create_provider("quickrouter", model="gpt-4o-mini")
        print("  Grader: GPT-4o-mini via QuickRouter")
    except Exception as e:
        print(f"  FATAL: Cannot create grader: {e}")
        return []

    # Determine what's already done
    done_keys = set()
    all_grades = []
    if resume_from:
        for entry in resume_from:
            key = (entry["model"], entry["task_type"], entry["problem_id"],
                   entry["generation"])
            done_keys.add(key)
            all_grades.append(entry)

    to_process = [e for e in lineage
                  if (e["model"], e["task_type"], e["problem_id"], e["generation"])
                  not in done_keys]

    print(f"  {len(to_process)} entries to grade, {len(done_keys)} already cached")

    for i, entry in enumerate(to_process):
        task_type = entry["task_type"]
        prompt_template = GRADE_PROMPTS.get(task_type)
        if not prompt_template:
            print(f"  [WARN] No grading prompt for {task_type}")
            continue

        prompt = prompt_template.format(
            problem=entry["problem_statement"],
            answer=entry["text"][:2000],
        )
        label = f"grade/{entry['model']}/{task_type}/{entry['problem_id']}_gen{entry['generation']}"

        try:
            resp = call_with_retry(grader, prompt, max_tokens=16, temperature=0.2,
                                   label=label, max_retries=3, base_delay=2.0)
            # Extract integer score
            nums = re.findall(r'\b(10|[1-9])\b', resp)
            if nums:
                score = int(nums[0])
                score = max(1, min(10, score))
            else:
                # Try to parse any number
                nums_any = re.findall(r'\d+', resp)
                score = int(nums_any[0]) if nums_any else 5
                score = max(1, min(10, score))
        except Exception as e:
            print(f"  [WARN] Grade failed for {label}: {e}")
            score = 5  # neutral fallback

        grade_entry = {
            "model": entry["model"],
            "task_type": entry["task_type"],
            "problem_id": entry["problem_id"],
            "generation": entry["generation"],
            "quality_score": score,
        }
        all_grades.append(grade_entry)
        done_keys.add((entry["model"], entry["task_type"], entry["problem_id"],
                      entry["generation"]))

        if (i + 1) % 15 == 0:
            print(f"  Processed {i+1}/{len(to_process)} entries...")
            save_checkpoint("phase3_grades.json", all_grades)

        time.sleep(0.5)

    save_checkpoint("phase3_grades.json", all_grades)
    print(f"  Phase 3 complete: {len(all_grades)} grades")
    return all_grades


# ---------------------------------------------------------------------------
# Phase 4: Correlation analysis
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float:
    n = len(values)
    return sum(values) / n if n > 0 else 0.0


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average tie handling."""
    def rank_vals(vals):
        sorted_pairs = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(sorted_pairs):
            j = i
            while j < len(sorted_pairs) and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j
        return ranks
    if len(xs) < 3:
        return 0.0
    return pearson_r(rank_vals(xs), rank_vals(ys))


def fit_exponential_beta(gens: list[int], c_values: list[float]) -> tuple[float, float]:
    """Fit exponential decay: C_k = C_0 * exp(-beta * k).

    Returns (beta, r_squared).
    """
    if len(gens) < 3:
        return 0.0, 0.0

    # log(C_k) = log(C_0) - beta * k
    log_c = [math.log(max(c, 1e-10)) for c in c_values]
    mx = mean(gens)
    my = mean(log_c)
    num = sum((g - mx) * (lc - my) for g, lc in zip(gens, log_c))
    den = sum((g - mx) ** 2 for g in gens)

    if den == 0:
        return 0.0, 0.0

    slope = num / den
    beta_val = -slope  # beta = -slope
    intercept = my - slope * mx

    ss_res = sum((lc - (intercept + slope * g)) ** 2 for g, lc in zip(gens, log_c))
    ss_tot = sum((lc - my) ** 2 for lc in log_c)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return max(beta_val, 0.0), r2


def run_phase4(extractions: list[dict], grades: list[dict]) -> dict:
    """Correlate constraint beta with quality degradation.

    Args:
        extractions: Phase 2 constraint extraction entries.
        grades: Phase 3 quality grading entries.

    Returns:
        Correlation results dict.
    """
    print("\n" + "=" * 70)
    print("  Phase 4: Correlate beta with Quality Degradation")
    print("=" * 70)

    # Build lookup: (model, task_type, problem_id, generation) -> total_constraint
    c_lookup: dict[tuple, float] = {}
    for e in extractions:
        key = (e["model"], e["task_type"], e["problem_id"], e["generation"])
        c_lookup[key] = e["total_constraint"]

    # Build lookup: (model, task_type, problem_id, generation) -> quality_score
    q_lookup: dict[tuple, float] = {}
    for g in grades:
        key = (g["model"], g["task_type"], g["problem_id"], g["generation"])
        q_lookup[key] = float(g["quality_score"])

    # Aggregate per (model, task_type) pair
    models = sorted(set(e["model"] for e in extractions))
    task_types = sorted(set(e["task_type"] for e in extractions))

    pair_results = []
    betas_all = []
    deltas_all = []
    by_task: dict[str, dict[str, list]] = {
        tt: {"betas": [], "deltas": [], "pair_names": []} for tt in task_types
    }

    for model in models:
        for tt in task_types:
            # Collect per-generation constraint and quality (average across problems)
            gen_constraints: dict[int, list[float]] = {}
            gen_qualities: dict[int, list[float]] = {}

            for e in extractions:
                if e["model"] == model and e["task_type"] == tt:
                    k = (model, tt, e["problem_id"], e["generation"])
                    gen = e["generation"]
                    if k in c_lookup:
                        gen_constraints.setdefault(gen, []).append(c_lookup[k])

            for g in grades:
                if g["model"] == model and g["task_type"] == tt:
                    k = (model, tt, g["problem_id"], g["generation"])
                    gen = g["generation"]
                    if k in q_lookup:
                        gen_qualities.setdefault(gen, []).append(q_lookup[k])

            if not gen_constraints or len(gen_constraints) < 3:
                print(f"  [{model}/{tt}] insufficient data, skipping")
                continue

            # Mean constraint per generation
            gens = sorted(gen_constraints.keys())
            c_means = [mean(gen_constraints[g]) for g in gens]

            # Fit beta
            beta, r2 = fit_exponential_beta(gens, c_means)

            # Quality degradation
            q_means = {g: mean(gen_qualities.get(g, [5.0])) for g in gens}
            gen1_q = q_means.get(1, 5.0)
            gen3_q = q_means.get(GENERATIONS, q_means.get(max(gens), 5.0))
            delta_q = gen3_q - gen1_q  # negative = degradation

            betas_all.append(beta)
            deltas_all.append(delta_q)
            by_task[tt]["betas"].append(beta)
            by_task[tt]["deltas"].append(delta_q)
            by_task[tt]["pair_names"].append(f"{model}/{tt}")

            pair_results.append({
                "model": model,
                "task_type": tt,
                "beta": round(beta, 6),
                "beta_r2": round(r2, 4),
                "constraint_per_gen": {str(g): round(c_means[i], 6)
                                       for i, g in enumerate(gens)},
                "quality_per_gen": {str(g): round(q_means[g], 4) for g in gens},
                "quality_degradation": round(delta_q, 4),
            })
            print(f"  {model:25s} {tt:20s}  beta={beta:.6f}  "
                  f"delta_q={delta_q:+.4f}  R²={r2:.4f}")

    # Overall correlation
    n = len(betas_all)
    if n >= 3:
        overall_r = pearson_r(betas_all, deltas_all)
        overall_rho = spearman_rho(betas_all, deltas_all)
    else:
        overall_r = 0.0
        overall_rho = 0.0

    print(f"\n  Overall (n={n}):  Pearson r = {overall_r:.4f}  Spearman rho = {overall_rho:.4f}")

    # Per-task-type correlations
    task_correlations = {}
    for tt in task_types:
        b = by_task[tt]["betas"]
        d = by_task[tt]["deltas"]
        if len(b) >= 3:
            task_correlations[tt] = {
                "n_pairs": len(b),
                "pearson_r": round(pearson_r(b, d), 4),
                "spearman_rho": round(spearman_rho(b, d), 4),
            }
            print(f"  {tt:25s} (n={len(b)}):  r = {task_correlations[tt]['pearson_r']:.4f}  "
                  f"rho = {task_correlations[tt]['spearman_rho']:.4f}")
        else:
            task_correlations[tt] = {
                "n_pairs": len(b),
                "pearson_r": None,
                "spearman_rho": None,
                "note": "Insufficient data (< 3 pairs)",
            }

    return {
        "overall": {
            "n_pairs": n,
            "pearson_r": round(overall_r, 4),
            "spearman_rho": round(overall_rho, 4),
        },
        "by_task_type": task_correlations,
        "per_pair": pair_results,
        "all_betas": [round(b, 6) for b in betas_all],
        "all_deltas": [round(d, 4) for d in deltas_all],
    }


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def load_checkpoint(filename: str) -> Optional[list]:
    """Load a cached phase result."""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded checkpoint: {filename} ({len(data)} entries)")
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Corrupt checkpoint {filename}: {e}")
    return None


def save_checkpoint(filename: str, data: list):
    """Save intermediate results atomically."""
    path = os.path.join(CACHE_DIR, filename)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [WARN] Failed to save checkpoint {filename}: {e}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary_table(pair_results: list[dict], correlation: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("  DOWNSTREAM VALIDATION v2 — RESULTS SUMMARY")
    print("=" * 90)

    # Table: per model-task pair
    header = f"  {'Model':<25s} {'Task':<20s} {'Beta':>10s} {'R²_fit':>8s} {'Delta_Q':>8s}  {'Q-Gen1':>7s} {'Q-Gen3':>7s}"
    print("\n" + header)
    print("  " + "-" * 90)

    for p in pair_results:
        q = p["quality_per_gen"]
        q1 = q.get("1", "-")
        q3 = q.get("3", q.get(str(max(int(k) for k in q.keys())), "-"))
        q1_str = f"{q1:.2f}" if isinstance(q1, (int, float)) else str(q1)
        q3_str = f"{q3:.2f}" if isinstance(q3, (int, float)) else str(q3)
        print(f"  {p['model']:<25s} {p['task_type']:<20s} "
              f"{p['beta']:>10.6f} {p['beta_r2']:>8.4f} {p['quality_degradation']:>+8.4f}  "
              f"{q1_str:>7s} {q3_str:>7s}")

    # Correlation summary
    print("\n  Correlation: Constraint Beta vs Quality Degradation")
    print("  " + "-" * 90)
    ov = correlation["overall"]
    print(f"  Overall (n={ov['n_pairs']}): "
          f"Pearson r = {ov['pearson_r']:.4f}, Spearman rho = {ov['spearman_rho']:.4f}")

    for tt, tc in correlation.get("by_task_type", {}).items():
        if tc.get("pearson_r") is not None:
            print(f"  {tt:25s} (n={tc['n_pairs']}): "
                  f"r = {tc['pearson_r']:.4f}, rho = {tc['spearman_rho']:.4f}")
        else:
            print(f"  {tt:25s} (n={tc['n_pairs']}): {tc.get('note', 'N/A')}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("  Downstream Validation v2 — Pure LLM-Judge Extractor")
    print(f"  Run at: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Models: {list(MODEL_CONFIGS.keys())}")
    print(f"  Task types: {TASK_TYPES}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Problems per task: 10")
    print("=" * 90)

    # ---- Phase 1: Generate Lineage ----
    lineage_cache = load_checkpoint("phase1_lineage.json")
    lineage = run_phase1(resume_from=lineage_cache)

    if not lineage:
        print("\n[FATAL] No lineage data generated. Exiting.")
        return

    # ---- Phase 2: Extract Constraints ----
    extraction_cache = load_checkpoint("phase2_extractions.json")
    extractions = extract_constraints(lineage, resume_from=extraction_cache)

    if not extractions:
        print("\n[FATAL] No constraint extractions. Exiting.")
        return

    # ---- Phase 3: Grade Quality ----
    grade_cache = load_checkpoint("phase3_grades.json")
    grades = grade_quality(lineage, resume_from=grade_cache)

    if not grades:
        print("\n[FATAL] No quality grades. Exiting.")
        return

    # ---- Phase 4: Correlation Analysis ----
    correlation = run_phase4(extractions, grades)

    # ---- Assemble Final Output ----
    output = {
        "metadata": {
            "script": "run_downstream_v2.py",
            "method": "pure_llm_judge_extractor",
            "extractor": "MultiJudgeExtractor (gpt-4o-mini, 1 repeat)",
            "grader": "GPT-4o-mini LLM judge (1-10 scale)",
            "generator_models": list(MODEL_CONFIGS.keys()),
            "task_types": TASK_TYPES,
            "n_problems_per_task": 10,
            "generations": GENERATIONS,
            "n_lineage_entries": len(lineage),
            "n_extractions": len(extractions),
            "n_grades": len(grades),
            "dimensions": DIMENSIONS,
            "run_at": datetime.now(timezone.utc).isoformat(),
        },
        "phase4_correlation": correlation,
    }

    # Save final results
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_PATH)
    print(f"\n  Final results saved to: {OUTPUT_PATH}")

    # Print summary
    print_summary_table(correlation.get("per_pair", []), correlation)

    print(f"\n  Cache files in: {CACHE_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
