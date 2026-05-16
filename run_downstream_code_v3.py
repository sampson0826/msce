#!/usr/bin/env python3
"""
Downstream Code Validation v3 -- Code-Focused beta -> Quality Mapping
======================================================================

Uses the EXISTING HybridConstraintExtractor (pure text features mode, no LLM judge)
to measure constraint beta for CODE tasks, then correlates beta with LLM-judged
code quality degradation across recursive "review this code" generations.

Why CODE-only:
  The hybrid extractor's text features (logic density, bigram repetition,
  proper-case ratio, etc.) produce meaningful beta for code because code has
  syntax checks, structural patterns, and compiler-level features that degrade
  measurably.  Fact/math tasks collapsed because the extractor can't distinguish
  correct-but-different phrasing from genuine degradation.

Design:
  - 35 Python coding problems (easy/medium LeetCode-style)
  - 5 models: GPT-4o-mini, DeepSeek-V3, DeepSeek-V4 Flash, Claude Sonnet 4.6, GPT-4o
  - 4 generations: Gen0=prompt, Gen1=generate, Gen2=review Gen1, Gen3=review Gen2
  - HybridConstraintExtractor(judge_fn=None) -- pure text features
  - GPT-4o-mini LLM judge for code quality (1-10: correctness, efficiency, readability)
  - Pearson r + Spearman rho: beta vs quality degradation slope

Output: experiment_data/downstream_code_v3_results.json
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

# Load .env
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

from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import (
    HybridConstraintExtractor,
    ConstraintState,
    _safe_float,
    extract_text_features,
    text_features_to_constraint,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_DATA = os.path.join(BASE_DIR, "experiment_data")
CACHE_DIR = os.path.join(EXPERIMENT_DATA, "downstream_code_v3_cache")
OUTPUT_PATH = os.path.join(EXPERIMENT_DATA, "downstream_code_v3_results.json")
os.makedirs(CACHE_DIR, exist_ok=True)

GENERATIONS = 4  # Gen1, Gen2, Gen3, Gen4 (after Gen0=prompt)

# ---------------------------------------------------------------------------
# Models under test
# ---------------------------------------------------------------------------
# DeepSeek-V3: deepseek-chat (during transition period, may map to V4 Flash)
# DeepSeek-V4 Flash: deepseek-v4-flash
MODELS = [
    {"name": "gpt-4o-mini",       "provider": "quickrouter", "model": "gpt-4o-mini"},
    {"name": "deepseek-v3",       "provider": "deepseek",    "model": "deepseek-chat"},
    {"name": "deepseek-v4-flash", "provider": "deepseek",    "model": "deepseek-v4-flash"},
    {"name": "claude-sonnet-4-6", "provider": "quickrouter", "model": "claude-sonnet-4-6"},
    {"name": "gpt-4o",            "provider": "quickrouter", "model": "gpt-4o"},
]

# ---------------------------------------------------------------------------
# 35 Python coding problems (easy/medium, LeetCode-style)
# ---------------------------------------------------------------------------
CODE_PROBLEMS = [
    # String manipulation (8)
    ("reverse_string", "Write a Python function `reverse_string(s)` that returns the reversed string. Do NOT use slicing s[::-1]."),
    ("is_palindrome", "Write a Python function `is_palindrome(s)` that returns True if string s is a palindrome (ignoring case and non-alphanumeric chars)."),
    ("count_vowels", "Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u) in string s, case-insensitive."),
    ("longest_word", "Write a Python function `longest_word(sentence)` that returns the longest word in a sentence. If there is a tie, return the first one."),
    ("valid_anagram", "Write a Python function `is_anagram(s1, s2)` that returns True if s1 and s2 are anagrams of each other."),
    ("length_of_last_word", "Write a Python function `length_of_last_word(s)` that returns the length of the last word in string s."),
    ("add_binary", "Write a Python function `add_binary(a, b)` that takes two binary strings a and b and returns their sum as a binary string."),
    ("first_unique_char", "Write a Python function `first_unique_char(s)` that returns the index of the first non-repeating character in s, or -1 if none exist."),

    # Arrays and numbers (10)
    ("two_sum", "Write a Python function `two_sum(nums, target)` that returns indices of the two numbers that add up to target. Assume exactly one solution."),
    ("max_subarray", "Write a Python function `max_subarray(nums)` that finds the contiguous subarray with the largest sum (Kadane's algorithm) and returns that sum."),
    ("move_zeroes", "Write a Python function `move_zeroes(nums)` that moves all 0's to the end of the list while maintaining the relative order of non-zero elements. Modify in-place."),
    ("remove_duplicates", "Write a Python function `remove_duplicates(nums)` that removes duplicates from a sorted list in-place and returns the new length."),
    ("rotate_array", "Write a Python function `rotate_array(nums, k)` that rotates the array to the right by k steps in-place."),
    ("single_number", "Write a Python function `single_number(nums)` where every element appears twice except one. Find that single one using O(n) time and O(1) space."),
    ("majority_element", "Write a Python function `majority_element(nums)` that returns the majority element (appears more than n/2 times)."),
    ("plus_one", "Write a Python function `plus_one(digits)` that takes a list of digits representing an integer and returns the list of digits after adding one."),
    ("intersection_two_arrays", "Write a Python function `intersection(nums1, nums2)` that returns the intersection of two arrays (unique elements that appear in both)."),
    ("missing_number", "Write a Python function `missing_number(nums)` that finds the missing number from 0 to n in an array containing n distinct numbers."),

    # Math / Number theory (7)
    ("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns a list of strings from 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number as string."),
    ("is_prime", "Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise."),
    ("factorial", "Write a Python function `factorial(n)` that returns n! (n factorial) computed iteratively (not recursively)."),
    ("fibonacci", "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1) using iteration."),
    ("sqrt_int", "Write a Python function `my_sqrt(x)` that returns the integer square root of a non-negative integer x without using math.sqrt or ** 0.5."),
    ("is_power_of_two", "Write a Python function `is_power_of_two(n)` that returns True if n is a power of two."),
    ("climbing_stairs", "Write a Python function `climbing_stairs(n)` that returns the number of distinct ways to climb n stairs taking 1 or 2 steps at a time."),

    # Data structures (5)
    ("valid_parentheses", "Write a Python function `is_valid_parentheses(s)` that returns True if the string s contains valid parentheses: '()', '[]', '{}' must be properly closed and nested."),
    ("merge_sorted_lists", "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list without using sorted() or .sort()."),
    ("group_anagrams", "Write a Python function `group_anagrams(words)` that groups a list of strings into lists of anagrams. Return a list of lists."),
    ("longest_common_prefix", "Write a Python function `longest_common_prefix(strs)` that finds the longest common prefix string among a list of strings."),
    ("roman_to_int", "Write a Python function `roman_to_int(s)` that converts a Roman numeral string (I, V, X, L, C, D, M) to an integer."),

    # Search / Misc (5)
    ("binary_search", "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted list arr, or -1 if not found."),
    ("str_str", "Write a Python function `str_str(haystack, needle)` that returns the index of the first occurrence of needle in haystack, or -1 if not found. Do NOT use .find() or .index()."),
    ("caesar_cipher", "Write a Python function `caesar_cipher(text, shift)` that applies a Caesar cipher to text with the given shift (0-25). Preserve case and skip non-letters."),
    ("count_primes", "Write a Python function `count_primes(n)` that returns the number of prime numbers less than n using the Sieve of Eratosthenes."),
    ("happy_number", "Write a Python function `is_happy(n)` that returns True if n is a happy number. A happy number eventually reaches 1 when replacing the number by the sum of squares of its digits."),
]

# ---------------------------------------------------------------------------
# Quality grading prompt (GPT-4o-mini as code judge)
# ---------------------------------------------------------------------------

GRADE_CODE_PROMPT = """You are an expert code evaluator. Score the code in the following text on a 1-10 scale.
Evaluate ONLY the code quality, not the surrounding commentary or review text.

Scoring criteria:
- 10: Perfect solution. Correct algorithm, handles edge cases, efficient, clean, readable code.
- 8-9: Correct solution with minor issues (slightly suboptimal, minor style issues).
- 6-7: Mostly correct logic but has bugs or missing edge cases, or significant inefficiency.
- 4-5: Partially correct approach but significant errors or very inefficient.
- 2-3: Some relevant code present but fundamentally wrong approach or mostly broken.
- 1: No working code, completely wrong, or nonsensical output.

Problem: {problem}

Text to evaluate:
---
{answer}
---

First, extract the main code/function from the text (ignore review commentary).
Then score it.

Reply with ONLY a single integer from 1 to 10:"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(values):
    n = len(values)
    return sum(values) / n if n > 0 else 0.0


def pearson_r(xs, ys):
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


def spearman_rho(xs, ys):
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


def fit_exponential_beta(generations, c_values):
    """Fit exponential decay: C_k = C_0 * (1-beta)^k  ->  beta, r_squared."""
    if len(generations) < 3:
        return None, 0.0
    gens = [float(g) for g in generations]
    log_c = [math.log(max(c, 1e-10)) for c in c_values]
    mx = mean(gens)
    my = mean(log_c)
    num = sum((g - mx) * (lc - my) for g, lc in zip(gens, log_c))
    den = sum((g - mx) ** 2 for g in gens)
    if den == 0:
        return None, 0.0
    slope = num / den
    # For decay (slope<0): C=C0*exp(-beta*k), beta = 1-exp(slope)
    # For growth (slope>0): C=C0*exp(+beta*k), beta = exp(slope)-1
    # Both capture per-generation degradation rate
    beta_val = abs(math.exp(slope) - 1.0)
    intercept = my - slope * mx
    ss_res = sum((lc - (intercept + slope * g)) ** 2 for g, lc in zip(gens, log_c))
    ss_tot = sum((lc - my) ** 2 for lc in log_c)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta_val, r2


def _pearson_p_value(r, n):
    """Two-tailed p-value for Pearson r via Fisher z-transformation."""
    if n <= 3:
        return 1.0
    if abs(r) >= 1.0 - 1e-15:
        return 0.0
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    z_stat = z * math.sqrt(n - 3)
    tail = 0.5 * math.erfc(abs(z_stat) / math.sqrt(2.0))
    return max(0.0, min(1.0, 2.0 * tail))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(filename):
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


def save_checkpoint(filename, data):
    path = os.path.join(CACHE_DIR, filename)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [WARN] Failed to save checkpoint {filename}: {e}")


# ---------------------------------------------------------------------------
# Robust API call with retry
# ---------------------------------------------------------------------------

def call_with_retry(adapter, prompt, max_tokens=512, temperature=0.7,
                     max_retries=4, base_delay=2.0, label=""):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = adapter.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if resp and len(resp.strip()) >= 1:
                return resp.strip()
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    [{label}] empty response, retry {attempt+1}/{max_retries} in {delay:.0f}s...")
                time.sleep(delay)
        except Exception as e:
            last_err = e
            err_str = str(e)[:150]
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if "429" in err_str or "rate" in err_str.lower():
                    delay = max(delay, 10.0)
                print(f"    [{label}] {type(e).__name__}: {err_str} -- retry {attempt+1}/{max_retries} in {delay:.0f}s")
                time.sleep(delay)
            else:
                print(f"    [{label}] FAILED after {max_retries} attempts: {type(e).__name__}: {err_str}")
    return "[ERROR]" if not last_err else f"[ERROR: {type(last_err).__name__}]"


# ---------------------------------------------------------------------------
# Phase 1: Generate recursive review lineage
# ---------------------------------------------------------------------------

def generate_lineage(model_cfg, problems, delay=1.0):
    """Generate recursive review lineage for one model on all code problems.

    Pipeline:
      Gen0 = problem prompt (not generated)
      Gen1 = model generates solution from problem
      Gen2 = model generates from "Review this code: [Gen1]"
      Gen3 = model generates from "Review this code: [Gen2]"
      Gen4 = model generates from "Review this code: [Gen3]"

    Returns list of {model, problem_id, generation, text, role}
    """
    model_name = model_cfg["name"]
    adapter = create_provider(model_cfg["provider"], model=model_cfg["model"])
    results = []
    n = len(problems)

    for pi, (prob_id, prob_stmt) in enumerate(problems):
        current_text = prob_stmt  # Start with the problem prompt
        for gen in range(1, GENERATIONS + 1):
            label = f"{model_name}/{prob_id}_gen{gen}"
            if gen == 1:
                prompt = current_text  # Gen1: solve the problem
            else:
                prompt = f"Review this code carefully. Identify any bugs, suggest improvements, and provide the corrected version:\n\n```\n{current_text}\n```"

            resp = call_with_retry(
                adapter, prompt, max_tokens=800, temperature=0.7, label=label,
            )
            results.append({
                "model": model_name,
                "problem_id": prob_id,
                "problem_statement": prob_stmt,
                "generation": gen,
                "text": resp,
                "role": "generate" if gen == 1 else "review",
            })
            current_text = resp  # Recursive: output -> next input
            time.sleep(delay)

        if (pi + 1) % 5 == 0 or pi == n - 1:
            print(f"  [{model_name}] {pi+1}/{n} problems done")

    return results


def run_phase1(resume_from=None):
    """Generate recursive review lineage for all models."""
    print("\n" + "=" * 70)
    print("  Phase 1: Generate Recursive Review Lineage (Code Only)")
    print("=" * 70)
    print(f"  Problems: {len(CODE_PROBLEMS)}, Generations: {GENERATIONS}")
    print(f"  Models: {[m['name'] for m in MODELS]}")

    done_keys = set()
    all_entries = []
    if resume_from:
        for entry in resume_from:
            done_keys.add((entry["model"], entry["problem_id"], entry["generation"]))
            all_entries.append(entry)

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        remaining = sum(1 for p in CODE_PROBLEMS
                        for g in range(1, GENERATIONS + 1)
                        if (model_name, p[0], g) not in done_keys)
        if remaining == 0:
            print(f"\n  [{model_name}] already complete, skipping")
            continue

        print(f"\n  [{model_name}] via {model_cfg['provider']}/{model_cfg['model']} -- {remaining} generations needed")

        try:
            entries = generate_lineage(model_cfg, CODE_PROBLEMS, delay=0.8)
            for e in entries:
                k = (e["model"], e["problem_id"], e["generation"])
                if k not in done_keys:
                    all_entries.append(e)
                    done_keys.add(k)
            save_checkpoint("phase1_lineage.json", all_entries)
            print(f"  [{model_name}] done. Checkpoint saved. ({len(all_entries)} total)")
        except Exception as e:
            print(f"  ERROR [{model_name}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            save_checkpoint("phase1_lineage.json", all_entries)

    print(f"\n  Phase 1 complete: {len(all_entries)} total lineage entries")
    return all_entries


# ---------------------------------------------------------------------------
# Phase 2: Extract constraint scores via HybridConstraintExtractor
# ---------------------------------------------------------------------------

def extract_constraints(lineage, resume_from=None):
    """Extract constraint scores for all lineage entries.

    Uses HybridConstraintExtractor(judge_fn=None) -- pure text features.
    No LLM judge needed: text features (logic density, bigram repetition,
    proper case ratio, etc.) produce meaningful signal for code degradation.
    """
    print("\n" + "=" * 70)
    print("  Phase 2: Extract Constraint Scores (Hybrid Extractor, pure text features)")
    print("=" * 70)

    extractor = HybridConstraintExtractor(judge_fn=None)

    done_keys = set()
    all_extractions = []
    if resume_from:
        for entry in resume_from:
            key = (entry["model"], entry["problem_id"], entry["generation"])
            done_keys.add(key)
            all_extractions.append(entry)

    to_process = [e for e in lineage
                  if (e["model"], e["problem_id"], e["generation"]) not in done_keys]
    print(f"  {len(to_process)} entries to extract, {len(done_keys)} already cached")

    for i, entry in enumerate(to_process):
        text = entry["text"]
        text_id = f"{entry['model']}/{entry['problem_id']}_gen{entry['generation']}"

        try:
            # Extract text features and map to ConstraintState
            features = extract_text_features(text)
            state = text_features_to_constraint(features)

            # Compute total constraint = ||sigma vector||
            sigma_vec = [
                state.sigma_fact,
                state.sigma_syntax,
                state.sigma_style,
                state.sigma_safety,
                state.sigma_coherence,
            ]
            total_constraint = float(math.sqrt(sum(v ** 2 for v in sigma_vec)))

            extraction_entry = {
                "model": entry["model"],
                "problem_id": entry["problem_id"],
                "generation": entry["generation"],
                "constraint_state": {
                    "sigma_fact": round(state.sigma_fact, 6),
                    "sigma_syntax": round(state.sigma_syntax, 6),
                    "sigma_style": round(state.sigma_style, 6),
                    "sigma_safety": round(state.sigma_safety, 6),
                    "sigma_coherence": round(state.sigma_coherence, 6),
                },
                "total_constraint": round(total_constraint, 6),
                "text_features": {
                    k: round(v, 6) for k, v in features.items()
                },
            }
            all_extractions.append(extraction_entry)
            done_keys.add((entry["model"], entry["problem_id"], entry["generation"]))

        except Exception as e:
            print(f"  [WARN] Extraction failed for {text_id}: {e}")
            extraction_entry = {
                "model": entry["model"],
                "problem_id": entry["problem_id"],
                "generation": entry["generation"],
                "constraint_state": {
                    "sigma_fact": 0.5, "sigma_syntax": 0.5,
                    "sigma_style": 0.5, "sigma_safety": 0.5, "sigma_coherence": 0.5,
                },
                "total_constraint": round(math.sqrt(5 * 0.25), 6),
                "text_features": {},
            }
            all_extractions.append(extraction_entry)

        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{len(to_process)} entries...")
            save_checkpoint("phase2_extractions.json", all_extractions)

    save_checkpoint("phase2_extractions.json", all_extractions)
    print(f"  Phase 2 complete: {len(all_extractions)} extractions")
    return all_extractions


# ---------------------------------------------------------------------------
# Phase 3: Grade code quality via GPT-4o-mini LLM judge
# ---------------------------------------------------------------------------

def grade_quality(lineage, resume_from=None):
    """Grade code quality for all lineage entries using GPT-4o-mini as judge.

    Returns list of {model, problem_id, generation, quality_score}.
    """
    print("\n" + "=" * 70)
    print("  Phase 3: Grade Code Quality (GPT-4o-mini LLM Judge, 1-10 scale)")
    print("=" * 70)

    try:
        grader = create_provider("quickrouter", model="gpt-4o-mini")
        print("  Grader: GPT-4o-mini via QuickRouter")
    except Exception as e:
        print(f"  FATAL: Cannot create grader: {e}")
        return []

    done_keys = set()
    all_grades = []
    if resume_from:
        for entry in resume_from:
            key = (entry["model"], entry["problem_id"], entry["generation"])
            done_keys.add(key)
            all_grades.append(entry)

    to_process = [e for e in lineage
                  if (e["model"], e["problem_id"], e["generation"]) not in done_keys]
    print(f"  {len(to_process)} entries to grade, {len(done_keys)} already cached")

    for i, entry in enumerate(to_process):
        prompt = GRADE_CODE_PROMPT.format(
            problem=entry["problem_statement"],
            answer=entry["text"][:2500],
        )
        label = f"grade/{entry['model']}/{entry['problem_id']}_gen{entry['generation']}"

        try:
            resp = call_with_retry(grader, prompt, max_tokens=16, temperature=0.1,
                                    label=label, max_retries=3, base_delay=2.0)
            nums = re.findall(r'\b(10|[1-9])\b', resp)
            if nums:
                score = int(nums[0])
                score = max(1, min(10, score))
            else:
                nums_any = re.findall(r'\d+', resp)
                score = int(nums_any[0]) if nums_any else 5
                score = max(1, min(10, score))
        except Exception as e:
            print(f"  [WARN] Grade failed for {label}: {e}")
            score = 5

        grade_entry = {
            "model": entry["model"],
            "problem_id": entry["problem_id"],
            "generation": entry["generation"],
            "quality_score": score,
        }
        all_grades.append(grade_entry)
        done_keys.add((entry["model"], entry["problem_id"], entry["generation"]))

        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{len(to_process)} entries...")
            save_checkpoint("phase3_grades.json", all_grades)

        time.sleep(0.3)

    save_checkpoint("phase3_grades.json", all_grades)
    print(f"  Phase 3 complete: {len(all_grades)} grades")
    return all_grades


# ---------------------------------------------------------------------------
# Phase 4: Correlation analysis
# ---------------------------------------------------------------------------

def compute_per_model_beta(extractions, model_name):
    """Compute beta for a single model from constraint data.

    Pools all problems, computes mean total_constraint per generation,
    fits exponential decay to get beta.
    """
    # Collect per-generation constraint values
    gen_constraints = {g: [] for g in range(1, GENERATIONS + 1)}
    for e in extractions:
        if e["model"] == model_name:
            gen = e["generation"]
            gen_constraints[gen].append(e["total_constraint"])

    # Mean per generation
    gens = sorted(gen_constraints.keys())
    c_means = [mean(gen_constraints[g]) for g in gens]

    if len(gens) < 3 or all(c < 1e-10 for c in c_means):
        return {"beta": None, "r2": 0.0, "c_means": c_means, "gens": gens}

    beta, r2 = fit_exponential_beta(gens, c_means)
    if beta is None:
        return {"beta": None, "r2": 0.0, "c_means": c_means, "gens": gens}

    return {
        "beta": round(beta, 6),
        "r2": round(r2, 4),
        "c_means": [round(c, 6) for c in c_means],
        "gens": gens,
    }


def compute_per_model_quality(grades, model_name):
    """Compute mean quality score per generation for a single model."""
    gen_qualities = {g: [] for g in range(1, GENERATIONS + 1)}
    for g in grades:
        if g["model"] == model_name:
            gen = g["generation"]
            gen_qualities[gen].append(g["quality_score"])

    gens = sorted(gen_qualities.keys())
    q_means = [mean(gen_qualities[g]) for g in gens]
    return {
        "gens": gens,
        "q_means": [round(q, 4) for q in q_means],
        "gen1_quality": round(q_means[0], 4) if q_means else None,
        "genN_quality": round(q_means[-1], 4) if q_means else None,
        "quality_slope": None,  # computed below
    }


def fit_quality_slope(gens, q_means):
    """Fit linear slope: quality = slope * generation + intercept.
    Returns slope (negative = quality degradation)."""
    if len(gens) < 2:
        return 0.0
    gens_f = [float(g) for g in gens]
    mx = mean(gens_f)
    my = mean(q_means)
    num = sum((g - mx) * (q - my) for g, q in zip(gens_f, q_means))
    den = sum((g - mx) ** 2 for g in gens_f)
    if den == 0:
        return 0.0
    return num / den


def run_phase4(extractions, grades):
    """Correlate constraint beta with code quality degradation across models.

    For each model:
      1. Compute beta from constraint decay across generations
      2. Compute quality slope across generations
      3. Correlate beta vs quality_slope across all models
    """
    print("\n" + "=" * 70)
    print("  Phase 4: Correlate beta with Code Quality Degradation")
    print("=" * 70)

    model_names = sorted(set(e["model"] for e in extractions))
    per_model = {}

    for model_name in model_names:
        beta_info = compute_per_model_beta(extractions, model_name)
        qual_info = compute_per_model_quality(grades, model_name)

        # Fit quality slope
        if qual_info["q_means"]:
            slope = fit_quality_slope(qual_info["gens"], qual_info["q_means"])
            qual_info["quality_slope"] = round(slope, 6)
            # quality_loss = -slope (positive = degradation)
            qual_info["quality_loss_per_gen"] = round(-slope, 6)

        per_model[model_name] = {
            "beta": beta_info.get("beta"),
            "beta_r2": beta_info.get("r2", 0.0),
            "constraint_per_gen": dict(zip(
                [str(g) for g in beta_info.get("gens", [])],
                beta_info.get("c_means", []),
            )),
            "quality_per_gen": dict(zip(
                [str(g) for g in qual_info.get("gens", [])],
                qual_info.get("q_means", []),
            )),
            "quality_slope": qual_info.get("quality_slope"),
            "quality_loss_per_gen": qual_info.get("quality_loss_per_gen"),
        }

        beta_str = f"{per_model[model_name]['beta']:.6f}" if per_model[model_name]['beta'] is not None else "N/A"
        slope_str = f"{per_model[model_name].get('quality_loss_per_gen', 0):+.4f}" if per_model[model_name].get('quality_loss_per_gen') is not None else "N/A"
        print(f"  {model_name:25s}  beta={beta_str:>10s}  "
              f"quality_loss/gen={slope_str:>10s}  "
              f"R2={per_model[model_name]['beta_r2']:.4f}")

    # Collect valid pairs
    valid_models = [m for m in model_names
                    if per_model[m]["beta"] is not None
                    and per_model[m]["quality_loss_per_gen"] is not None]
    betas = [per_model[m]["beta"] for m in valid_models]
    deltas = [per_model[m]["quality_loss_per_gen"] for m in valid_models]

    n = len(betas)
    if n >= 3:
        r = pearson_r(betas, deltas)
        rho = spearman_rho(betas, deltas)
        p_val = _pearson_p_value(r, n)
    else:
        r, rho, p_val = 0.0, 0.0, 1.0

    # Also compute per-generation averages across all models
    all_betas_found = sum(1 for m in model_names if per_model[m]["beta"] is not None)

    print(f"\n  Correlation (n={n} valid / {len(model_names)} total models):")
    print(f"    Pearson r  = {r:.4f}")
    print(f"    Spearman rho = {rho:.4f}")
    print(f"    p-value (Pearson) = {p_val:.4f}")
    if p_val < 0.05:
        print(f"    STATISTICALLY SIGNIFICANT at p < 0.05")
    else:
        print(f"    NOT significant at p < 0.05 (need more models or larger effect)")

    # Per-generation quality summary
    gen_quality_summary = {}
    for gen in range(1, GENERATIONS + 1):
        scores = [g["quality_score"] for g in grades if g["generation"] == gen]
        if scores:
            gen_quality_summary[str(gen)] = {
                "mean": round(mean(scores), 4),
                "min": min(scores),
                "max": max(scores),
                "n": len(scores),
            }

    return {
        "per_model": per_model,
        "correlation": {
            "n_valid_pairs": n,
            "n_models_total": len(model_names),
            "n_betas_found": all_betas_found,
            "pearson_r": round(r, 4),
            "spearman_rho": round(rho, 4),
            "p_value": round(p_val, 4),
            "significant_at_0.05": p_val < 0.05,
            "betas": [round(b, 6) for b in betas],
            "quality_losses": [round(d, 6) for d in deltas],
            "valid_model_names": valid_models,
        },
        "gen_quality_summary": gen_quality_summary,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary_table(phase4, grades, extractions):
    """Print formatted results summary."""
    per_model = phase4["per_model"]
    corr = phase4["correlation"]

    print("\n" + "=" * 90)
    print("  DOWNSTREAM CODE VALIDATION v3 -- RESULTS SUMMARY")
    print("=" * 90)

    # Per-model table
    header = (f"  {'Model':<25s} {'Beta':>10s} {'R2_fit':>8s} "
              f"{'Q-Gen1':>8s} {'Q-Gen4':>8s} {'Q-Loss/Gen':>11s}")
    print("\n" + header)
    print("  " + "-" * 90)
    for model_name, info in sorted(per_model.items()):
        beta_str = f"{info['beta']:.6f}" if info["beta"] is not None else "N/A"
        qp = info.get("quality_per_gen", {})
        q1 = qp.get("1", qp.get(str(min(int(k) for k in qp.keys()) if qp else 0), "-"))
        q4 = qp.get("4", qp.get(str(max(int(k) for k in qp.keys()) if qp else 0), "-"))
        q1_str = f"{q1:.2f}" if isinstance(q1, (int, float)) else str(q1)
        q4_str = f"{q4:.2f}" if isinstance(q4, (int, float)) else str(q4)
        loss_str = f"{info.get('quality_loss_per_gen', 0):+.4f}" if info.get('quality_loss_per_gen') is not None else "N/A"
        print(f"  {model_name:<25s} {beta_str:>10s} {info['beta_r2']:>8.4f} "
              f"{q1_str:>8s} {q4_str:>8s} {loss_str:>11s}")

    # Quality by generation
    print("\n  Quality by Generation (all models pooled):")
    gqs = phase4.get("gen_quality_summary", {})
    for gen in sorted(gqs.keys(), key=int):
        gq = gqs[gen]
        print(f"    Gen{gen}: mean={gq['mean']:.4f}  range=[{gq['min']}, {gq['max']}]  n={gq['n']}")

    # Correlation
    print(f"\n  Correlation: Beta vs Quality Loss/Gen")
    print(f"    n = {corr['n_valid_pairs']} valid pairs (of {corr['n_models_total']} models tested)")
    print(f"    Pearson r  = {corr['pearson_r']:.4f}")
    print(f"    Spearman rho = {corr['spearman_rho']:.4f}")
    print(f"    p-value = {corr['p_value']:.4f}")
    if corr["significant_at_0.05"]:
        print(f"    *** STATISTICALLY SIGNIFICANT at p < 0.05 ***")
    else:
        print(f"    Not significant at p < 0.05")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("  Downstream Code Validation v3 -- Code-Focused beta -> Quality")
    print(f"  Run at: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Models: {[m['name'] for m in MODELS]}")
    print(f"  Problems: {len(CODE_PROBLEMS)} code problems")
    print(f"  Generations: {GENERATIONS} (Gen1=solve, Gen2-4=review chain)")
    print(f"  Extractor: HybridConstraintExtractor (pure text features, no LLM judge)")
    print(f"  Grader: GPT-4o-mini LLM judge (1-10 scale)")
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
    phase4 = run_phase4(extractions, grades)

    # ---- Assemble Final Output ----
    # Count per-model generation entries
    model_gen_counts = {}
    for e in lineage:
        model_gen_counts.setdefault(e["model"], {})
        model_gen_counts[e["model"]].setdefault(str(e["generation"]), 0)
        model_gen_counts[e["model"]][str(e["generation"])] += 1

    output = {
        "metadata": {
            "script": "run_downstream_code_v3.py",
            "method": "hybrid_extractor_text_features_only",
            "extractor": "HybridConstraintExtractor (judge_fn=None, pure text features)",
            "grader": "GPT-4o-mini LLM judge (1-10: correctness, efficiency, readability)",
            "models": [m["name"] for m in MODELS],
            "model_configs": MODELS,
            "n_problems": len(CODE_PROBLEMS),
            "problem_ids": [p[0] for p in CODE_PROBLEMS],
            "generations": GENERATIONS,
            "generation_roles": ["gen1=generate", "gen2=review_gen1", "gen3=review_gen2", "gen4=review_gen3"],
            "n_lineage_entries": len(lineage),
            "n_extractions": len(extractions),
            "n_grades": len(grades),
            "model_gen_counts": model_gen_counts,
            "run_at": datetime.now(timezone.utc).isoformat(),
        },
        "phase4_results": phase4,
    }

    # Save final results
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_PATH)
    print(f"\n  Final results saved to: {OUTPUT_PATH}")

    # Print summary
    print_summary_table(phase4, grades, extractions)

    print(f"\n  Cache files in: {CACHE_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
