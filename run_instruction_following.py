#!/usr/bin/env python3
"""Instruction-Following Control Experiment for StabilityBench (Baseline #7).

Disentangles instruction decay (topic + format drift) from constraint-structure
degradation by running recursive generation for 3 models x 20 seeds x 4 gens,
then measuring topic fidelity (TF-IDF cosine vs Gen0) and format fidelity
(rubric-based binary checks per capability).

Output: experiment_data/instruction_following_results-2026-05-17.json
"""
import json
import os
import re
import sys
import time
import math
import numpy as np
from collections import defaultdict
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Load .env
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_PATH = os.path.join(BASE_DIR, "experiment_data", "n100",
                         "gpt-4o-mini_s100_lineage.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "experiment_data",
                           "instruction_following_results-2026-05-17.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "experiment_data",
                               "instruction_following_checkpoint-2026-05-17.json")

CAPABILITIES = ["math_reasoning", "code_generation", "factual_knowledge",
                "creative_writing"]
N_SEEDS_PER_CAP = 5         # 5 seeds per capability -> 20 total
N_GENERATIONS = 4            # Gen0, Gen1, Gen2, Gen3
MAX_TOKENS = 512
TEMPERATURE = 0.8
API_DELAY = 3.0              # seconds between API calls

MODELS = [
    {"model": "gpt-5.5",          "provider": "quickrouter",
     "name": "gpt-5.5_instr_ctrl",
     "constraint_beta": 0.01087094723806383},
    {"model": "gpt-4o-mini",      "provider": "quickrouter",
     "name": "gpt-4o-mini_instr_ctrl",
     "constraint_beta": 0.08848613155619972},
    {"model": "claude-opus-4-7",  "provider": "quickrouter",
     "name": "claude-opus-4-7_instr_ctrl",
     "constraint_beta": 0.16353372352921397},
]

# Model display name mapping for constraint beta lookup
CONSTRAINT_BETA_MAP = {
    "gpt-5.5": 0.01087094723806383,
    "gpt-4o-mini": 0.08848613155619972,
    "claude-opus-4-7": 0.16353372352921397,
}

# ── TF-IDF utilities (fallback when sentence-transformers unavailable) ─────────

def tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def compute_tf(terms: list[str]) -> dict[str, float]:
    """Term frequency normalized by doc length."""
    tf = {}
    n = len(terms)
    if n == 0:
        return tf
    for t in terms:
        tf[t] = tf.get(t, 0.0) + 1.0 / n
    return tf


def compute_idf(docs: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency."""
    N = len(docs)
    df = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    idf = {}
    for term, count in df.items():
        idf[term] = math.log((N + 1) / (count + 1)) + 1.0
    return idf


def tfidf_vector(terms: list[str], idf: dict[str, float]) -> np.ndarray:
    """Build a sparse-like TF-IDF vector for cosine similarity.

    Returns a dense vector over the vocabulary (IDF keys), which is fine for
    short texts and small doc sets.
    """
    tf = compute_tf(terms)
    vocab = sorted(idf.keys())
    vec = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        vec[i] = tf.get(term, 0.0) * idf.get(term, 0.0)
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Seed loading ───────────────────────────────────────────────────────────────

def load_seeds() -> list[dict]:
    """Load and stratify seeds: 5 per capability from the gpt-4o-mini lineage.

    Returns list of dicts with keys: id, text, capability.
    """
    entries = []
    with open(SEED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("generation") != 0:
                continue
            entries.append(e)

    # Group by capability
    by_cap = defaultdict(list)
    for e in entries:
        for tag in e.get("capability_tags", []):
            if tag in CAPABILITIES:
                by_cap[tag].append(e)
                break  # first matching capability

    seeds = []
    for cap in CAPABILITIES:
        cap_entries = by_cap.get(cap, [])
        if len(cap_entries) < N_SEEDS_PER_CAP:
            print(f"  [WARN] Only {len(cap_entries)} seeds for {cap}, "
                  f"need {N_SEEDS_PER_CAP}")
        selected = cap_entries[:N_SEEDS_PER_CAP]
        for e in selected:
            seeds.append({
                "id": e["id"],
                "text": e["text"],
                "capability": cap,
            })

    print(f"Loaded {len(seeds)} seeds across {len(CAPABILITIES)} capabilities:")
    for cap in CAPABILITIES:
        n = sum(1 for s in seeds if s["capability"] == cap)
        print(f"  {cap}: {n}")
    return seeds


# ── Format fidelity rubrics ────────────────────────────────────────────────────

def check_math_format(text: str) -> dict:
    """Check if text follows math reasoning format.

    Expected: numbered steps or bullet points, final answer in \\boxed{} or
    "Answer:" marker.
    """
    has_steps = bool(re.search(r'(?:Step\s*\d|^\s*\d+[\.\)]\s|\n\s*\d+[\.\)]\s)',
                               text))
    has_bullets = bool(re.search(r'[\n\r](?:\s*[-•*]\s)', text))
    has_boxed = "\\boxed{" in text or "\\boxed " in text
    has_answer_marker = bool(re.search(r'(?:Answer\s*:|Final answer\s*:|'
                                       r'Therefore,?\s+the\s+answer\s+is)',
                                       text, re.IGNORECASE))

    structured = has_steps or has_bullets
    final_answer = has_boxed or has_answer_marker

    return {
        "has_steps_or_bullets": structured,
        "has_final_answer_marker": final_answer,
        "format_ok": structured,  # primary check: structured derivation
    }


def check_code_format(text: str) -> dict:
    """Check if text follows code generation format.

    Expected: function definition and docstring/comment block.
    """
    has_function = bool(re.search(r'\bdef\s+\w+|function\s+\w+|'
                                  r'func\s+\w+\s*\(|'
                                  r'public\s+(?:static\s+)?\w+\s+\w+\s*\(',
                                  text))
    # Docstring: triple-quoted block or /* */ block comment near function
    has_docstring = bool(re.search(r'("""|\'\'\'|/\*\*|###\s|//\s*\w)',
                                   text))
    has_code_block = "```" in text

    return {
        "has_function_def": has_function,
        "has_docstring_or_comment": has_docstring,
        "has_code_block": has_code_block,
        "format_ok": has_function,  # primary check: function definition
    }


def check_factual_format(text: str) -> dict:
    """Check if text follows factual knowledge format.

    Expected: paragraph-form prose, >50 words, not bullet points or code.
    """
    words = text.split()
    word_count = len(words)
    is_prose = word_count >= 50
    # Not primarily bullet points
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    bullet_lines = sum(1 for l in lines if re.match(r'^[-•*\d+\.\)]\s', l))
    is_bullet_heavy = (bullet_lines / max(1, len(lines))) > 0.5
    # Not a code block
    code_block_ratio = text.count("```") / max(1, len(lines))

    return {
        "word_count": word_count,
        "is_prose": is_prose,
        "is_bullet_heavy": is_bullet_heavy,
        "format_ok": is_prose and not is_bullet_heavy and code_block_ratio < 0.3,
    }


def check_creative_format(text: str) -> dict:
    """Check if text follows creative writing format.

    Expected: narrative prose, >100 words, not meta-commentary.
    """
    words = text.split()
    word_count = len(words)
    is_narrative = word_count >= 100
    # Check for meta-commentary patterns
    meta_patterns = [
        r"Here(?:'s| is) a (?:story|poem|narrative)",
        r"I (?:hope|think) (?:this|you)",
        r"(?:Sure|Of course|Certainly)[,!]?\s+(?:here|I)",
        r"Let me (?:write|create|tell|craft)",
        r"This (?:story|poem|piece|narrative) (?:is about|explores)",
    ]
    has_meta = any(re.search(p, text, re.IGNORECASE) for p in meta_patterns)
    # Check for narrative structure indicators
    has_dialogue = bool(re.search(r'"([^"]+)"', text))
    has_narrative_elements = bool(re.search(
        r'(?:said|asked|replied|walked|looked|felt|thought|'
        r'suddenly|eventually|after|before|while|meanwhile)',
        text, re.IGNORECASE))

    return {
        "word_count": word_count,
        "is_narrative_length": is_narrative,
        "has_meta_commentary": has_meta,
        "has_dialogue": has_dialogue,
        "has_narrative_elements": has_narrative_elements,
        "format_ok": is_narrative and not has_meta,
    }


FORMAT_CHECKERS = {
    "math_reasoning": check_math_format,
    "code_generation": check_code_format,
    "factual_knowledge": check_factual_format,
    "creative_writing": check_creative_format,
}


def compute_format_fidelity(text: str, capability: str) -> tuple[bool, dict]:
    """Return (passes_format, rubric_results)."""
    checker = FORMAT_CHECKERS.get(capability)
    if checker is None:
        return False, {"error": f"no checker for {capability}"}
    result = checker(text)
    return result["format_ok"], result


# ── Topic fidelity (TF-IDF cosine similarity) ──────────────────────────────────

class TopicFidelityComputer:
    """Computes cosine similarity between generated texts and Gen0 prompts.

    Uses TF-IDF vectors over the corpus of all texts in the benchmark.
    """

    def __init__(self):
        self.idf = None
        self.gen0_texts: dict[str, str] = {}  # seed_id -> gen0 text

    def fit_idf(self, all_texts: list[str]):
        """Compute IDF from the full corpus of all generations."""
        tokenized = [tokenize(t) for t in all_texts if t]
        self.idf = compute_idf(tokenized)

    def set_gen0(self, seed_id: str, text: str):
        self.gen0_texts[seed_id] = text

    def compute(self, seed_id: str, gen_text: str) -> float:
        """Cosine similarity between gen_text and Gen0 for this seed."""
        if seed_id not in self.gen0_texts:
            return 0.0
        gen0_text = self.gen0_texts[seed_id]
        if self.idf is None:
            # Fit on the fly with just these two texts
            docs = [tokenize(gen0_text), tokenize(gen_text)]
            self.idf = compute_idf(docs)
        terms_gen0 = tokenize(gen0_text)
        terms_gen = tokenize(gen_text)
        vec0 = tfidf_vector(terms_gen0, self.idf)
        vec1 = tfidf_vector(terms_gen, self.idf)
        return cosine_similarity(vec0, vec1)


# ── Exponential decay fitting ──────────────────────────────────────────────────

def fit_exponential_decay(
    trajectory: list[float],
    gens: list[int],
) -> tuple[Optional[float], Optional[float]]:
    """Fit I_k = I_0 * exp(-beta * k) using OLS on log values.

    Args:
        trajectory: fidelity values at each generation (including Gen0=1.0).
        gens: generation indices (typically [0, 1, 2, 3]).

    Returns:
        (beta, r_squared) or (None, None) if fit fails.
    """
    if len(trajectory) < 3:
        return None, None
    # OLS on log(I_k) = log(I_0) - beta * k
    # Remove Gen0 (k=0) from fit since it's 1.0 by construction
    fit_gens = np.array([g for g in gens if g > 0], dtype=float)
    fit_vals = np.array([trajectory[gens.index(g)]
                         for g in gens if g > 0], dtype=float)

    if len(fit_gens) < 2:
        return None, None

    # Guard against zero/negative values
    mask = fit_vals > 1e-10
    fit_gens = fit_gens[mask]
    fit_vals = fit_vals[mask]

    if len(fit_gens) < 2:
        return None, None

    log_vals = np.log(fit_vals)

    # OLS: y = a + b*x, where b = -beta
    n = len(fit_gens)
    x_mean = np.mean(fit_gens)
    y_mean = np.mean(log_vals)
    ss_xy = np.sum((fit_gens - x_mean) * (log_vals - y_mean))
    ss_xx = np.sum((fit_gens - x_mean) ** 2)

    if ss_xx < 1e-10:
        return None, None

    b = ss_xy / ss_xx
    a = y_mean - b * x_mean

    beta = float(-b)  # b is negative (decay), so beta is positive
    # Clamp to reasonable range
    if beta < 0:
        beta = 0.0

    # R-squared
    y_pred = a + b * fit_gens
    ss_res = np.sum((log_vals - y_pred) ** 2)
    ss_tot = np.sum((log_vals - y_mean) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    return beta, r_squared


# ── Recursive generation ───────────────────────────────────────────────────────

def run_recursive_generation(
    seeds: list[dict],
    model_cfg: dict,
    checkpoint: Optional[dict] = None,
) -> dict:
    """Run recursive generation for one model.

    Returns dict with keys: generations (per-seed per-gen data), fidelity results.
    """
    model_name = model_cfg["name"]
    provider = model_cfg["provider"]
    model_id = model_cfg["model"]

    print(f"\n{'=' * 70}")
    print(f"Model: {model_name} ({model_id} via {provider})")
    print(f"{'=' * 70}")

    adapter = create_provider(provider, model=model_id,
                              temperature=TEMPERATURE,
                              max_tokens=MAX_TOKENS,
                              timeout_sec=120)

    # Check for checkpoint
    generations: dict[str, dict[int, str]] = {}  # seed_id -> {gen: text}
    completed_seeds = set()

    if checkpoint:
        cp_gens = checkpoint.get("generations", {})
        for sid, gens in cp_gens.items():
            if str(3) in gens or len(gens) >= 3:  # has Gen1-3
                generations[sid] = {int(k): v for k, v in gens.items()}
                completed_seeds.add(sid)

    if completed_seeds:
        print(f"  Resuming from checkpoint: {len(completed_seeds)} seeds "
              f"already complete")

    for i, seed in enumerate(seeds):
        sid = seed["id"]
        capability = seed["capability"]
        gen0_text = seed["text"]

        if sid in completed_seeds:
            continue

        # Gen0 is the original seed
        generations[sid] = {0: gen0_text}
        prev_text = gen0_text

        for gen in range(1, N_GENERATIONS):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(prev_text, max_tokens=MAX_TOKENS)
                    if resp and resp.strip():
                        break
                except Exception as e:
                    err_name = type(e).__name__
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        print(f"  [RETRY] {sid} Gen{gen} attempt {attempt+1}: "
                              f"{err_name} (wait {wait}s)")
                        time.sleep(wait)
                    else:
                        print(f"  [FAIL] {sid} Gen{gen}: {err_name}")
                        resp = "[EMPTY]"

            if not resp or not resp.strip():
                resp = "[EMPTY]"

            generations[sid][gen] = resp
            prev_text = resp

            # Small delay between generations within a seed
            time.sleep(0.5)

        # API delay between seeds
        time.sleep(API_DELAY)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{len(seeds)} seeds")

    # Count completed generations
    total_gen = sum(len(gens) for gens in generations.values())
    print(f"  Total generations: {total_gen} (across {len(generations)} seeds)")

    return {"generations": generations}


# ── Fidelity computation ───────────────────────────────────────────────────────

def compute_fidelity(
    seeds: list[dict],
    generations: dict[str, dict[int, str]],
) -> dict:
    """Compute topic and format fidelity for all seeds and generations."""
    seed_map = {s["id"]: s for s in seeds}

    # Collect all texts for IDF computation
    all_texts = []
    for sid, gens in generations.items():
        for gen, text in gens.items():
            if text and text.strip() and text != "[EMPTY]":
                all_texts.append(text)

    # Compute IDF once over the full corpus
    tf_computer = TopicFidelityComputer()
    tf_computer.fit_idf(all_texts)

    # Set Gen0 texts
    for sid, gens in generations.items():
        if 0 in gens:
            tf_computer.set_gen0(sid, gens[0])

    # Per-seed, per-generation fidelity
    per_seed_topic = defaultdict(dict)   # seed_id -> {gen: topic_fidelity}
    per_seed_format = defaultdict(dict)   # seed_id -> {gen: format_passes}
    per_seed_format_detail = defaultdict(dict)  # seed_id -> {gen: rubric_detail}
    per_seed_composite = defaultdict(dict)  # seed_id -> {gen: I_k}

    n_empty = 0
    n_seeds = len(seeds)

    for seed in seeds:
        sid = seed["id"]
        capability = seed["capability"]
        gens = generations.get(sid, {})

        for gen in sorted(gens.keys()):
            text = gens[gen]

            # Handle empty/failed generations
            if not text or text.strip() == "[EMPTY]" or len(text.strip()) == 0:
                n_empty += 1
                per_seed_topic[sid][gen] = None
                per_seed_format[sid][gen] = False
                per_seed_format_detail[sid][gen] = {"error": "empty generation"}
                per_seed_composite[sid][gen] = None
                continue

            # Topic fidelity
            if gen == 0:
                topic_fid = 1.0  # self-similarity
            else:
                topic_fid = tf_computer.compute(sid, text)

            # Handle NaN (e.g., empty vocabulary)
            if math.isnan(topic_fid):
                topic_fid = 0.0

            per_seed_topic[sid][gen] = topic_fid

            # Format fidelity
            if gen == 0:
                format_ok = True  # seeds are well-formed by construction
                format_detail = {"format_ok": True, "note": "Gen0 seed"}
            else:
                format_ok, format_detail = compute_format_fidelity(text,
                                                                   capability)
            per_seed_format[sid][gen] = format_ok
            per_seed_format_detail[sid][gen] = format_detail

            # Composite
            if topic_fid is not None and topic_fid >= 0:
                per_seed_composite[sid][gen] = topic_fid * (1.0 if format_ok
                                                            else 0.0)
            else:
                per_seed_composite[sid][gen] = None

    # Aggregate per generation
    gens_list = list(range(N_GENERATIONS))
    agg_topic = {}   # gen -> mean topic fidelity
    agg_format = {}  # gen -> fraction maintaining format
    agg_composite = {}  # gen -> mean composite

    for gen in gens_list:
        topics = [per_seed_topic[sid].get(gen) for sid in
                  [s["id"] for s in seeds]]
        topics = [t for t in topics if t is not None]
        agg_topic[gen] = float(np.mean(topics)) if topics else None

        formats = [per_seed_format[sid].get(gen) for sid in
                   [s["id"] for s in seeds]]
        formats_ok = [f for f in formats if isinstance(f, bool)]
        agg_format[gen] = float(np.mean(formats_ok)) if formats_ok else None

        composites = [per_seed_composite[sid].get(gen) for sid in
                      [s["id"] for s in seeds]]
        composites = [c for c in composites if c is not None]
        agg_composite[gen] = float(np.mean(composites)) if composites else None

    # Warn about empty generations
    warnings = []
    if n_empty > 0:
        pct = 100 * n_empty / (n_seeds * N_GENERATIONS)
        warnings.append(f"{n_empty} empty generations ({pct:.1f}%)")
        if pct > 20:
            warnings.append("CRITICAL: >20% degraded seeds, "
                            "beta_instruction marked unreliable")

    return {
        "per_seed_topic": {k: dict(v) for k, v in per_seed_topic.items()},
        "per_seed_format": {k: dict(v) for k, v in per_seed_format.items()},
        "per_seed_format_detail": {k: {str(g): d for g, d in v.items()}
                                    for k, v in per_seed_format_detail.items()},
        "per_seed_composite": {k: dict(v) for k, v in per_seed_composite.items()},
        "agg_topic": agg_topic,
        "agg_format": agg_format,
        "agg_composite": agg_composite,
        "n_empty": n_empty,
        "warnings": warnings,
    }


# ── Bootstrap confidence intervals ─────────────────────────────────────────────

def bootstrap_beta_ci(
    per_seed_composite: dict[str, dict[int, float]],
    n_bootstrap: int = 1000,
) -> dict:
    """Bootstrap CI for beta_instruction from per-seed composite trajectories.

    Returns dict with keys: beta_mean, ci_lower, ci_upper, bootstrap_betas.
    """
    # Extract per-seed beta estimates
    seed_betas = []
    for sid, comp in per_seed_composite.items():
        gens = sorted([int(k) for k in comp.keys()])
        traj = [comp[g] for g in gens if comp[g] is not None]
        if len(traj) >= 3:
            beta, _ = fit_exponential_decay(traj, gens[:len(traj)])
            if beta is not None:
                seed_betas.append(beta)

    if len(seed_betas) < 2:
        return {"beta_mean": None, "ci_lower": None, "ci_upper": None,
                "bootstrap_betas": [], "n_seeds_with_beta": len(seed_betas)}

    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(seed_betas, size=len(seed_betas), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    bootstrap_means = sorted(bootstrap_means)
    ci_lower = bootstrap_means[int(0.025 * n_bootstrap)]
    ci_upper = bootstrap_means[int(0.975 * n_bootstrap)]

    return {
        "beta_mean": float(np.mean(seed_betas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_betas": bootstrap_means,
        "n_seeds_with_beta": len(seed_betas),
        "per_seed_betas": seed_betas,
    }


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def save_checkpoint(data: dict):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint() -> Optional[dict]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("INSTRUCTION-FOLLOWING CONTROL EXPERIMENT (Baseline #7)")
    print("StabilityBench: Disentangling instruction decay from constraint decay")
    print("=" * 70)
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Seeds: {N_SEEDS_PER_CAP} per capability x {len(CAPABILITIES)} caps "
          f"= {N_SEEDS_PER_CAP * len(CAPABILITIES)} total")
    print(f"Generations: {N_GENERATIONS} (Gen0..Gen{N_GENERATIONS-1})")
    print(f"Total API calls: {N_SEEDS_PER_CAP * len(CAPABILITIES) * (N_GENERATIONS - 1) * len(MODELS)}")
    print(f"Embedding: TF-IDF cosine similarity")
    print(f"Format rubrics: regex/rule-based per capability")
    print()

    # ── Load seeds ──────────────────────────────────────────────────────────
    seeds = load_seeds()

    # ── Load or initialize checkpoint ───────────────────────────────────────
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\nLoaded checkpoint with {len(checkpoint.get('model_results', {}))} "
              f"completed models")

    model_results = checkpoint.get("model_results", {}) if checkpoint else {}

    # ── Run each model ──────────────────────────────────────────────────────
    for model_cfg in MODELS:
        model_key = model_cfg["name"]

        if model_key in model_results:
            print(f"\n[SKIP] {model_key} already in checkpoint, reusing data")
            continue

        # Run recursive generation
        gen_result = run_recursive_generation(
            seeds, model_cfg,
            checkpoint=model_results.get(model_key),
        )

        # Compute fidelity
        print(f"\nComputing fidelity for {model_key}...")
        fidelity = compute_fidelity(seeds, gen_result["generations"])

        # Fit exponential decay models
        gens = list(range(N_GENERATIONS))

        # Topic beta
        topic_traj = [fidelity["agg_topic"].get(g, 0.0) or 0.0
                      for g in gens]
        beta_topic, r2_topic = fit_exponential_decay(topic_traj, gens)

        # Format beta
        format_traj = [fidelity["agg_format"].get(g, 0.0) or 0.0
                       for g in gens]
        beta_format, r2_format = fit_exponential_decay(format_traj, gens)

        # Composite beta
        composite_traj = [fidelity["agg_composite"].get(g, 0.0) or 0.0
                          for g in gens]
        beta_instruction, r2_instr = fit_exponential_decay(composite_traj, gens)

        # Bootstrap CI for beta_instruction
        bootstrap_result = bootstrap_beta_ci(fidelity["per_seed_composite"])

        # Assemble result
        result = {
            "model_name": model_key,
            "model_id": model_cfg["model"],
            "provider": model_cfg["provider"],
            "constraint_beta": model_cfg["constraint_beta"],
            "n_seeds": len(seeds),
            "generations": N_GENERATIONS,
            "fidelity": {
                "agg_topic": fidelity["agg_topic"],
                "agg_format": fidelity["agg_format"],
                "agg_composite": fidelity["agg_composite"],
                "per_seed_topic": fidelity["per_seed_topic"],
                "per_seed_format": fidelity["per_seed_format"],
                "per_seed_format_detail": fidelity["per_seed_format_detail"],
                "per_seed_composite": fidelity["per_seed_composite"],
            },
            "beta_topic": beta_topic,
            "beta_topic_r2": r2_topic,
            "beta_format": beta_format,
            "beta_format_r2": r2_format,
            "beta_instruction": beta_instruction,
            "beta_instruction_r2": r2_instr,
            "beta_instruction_bootstrap": bootstrap_result,
            "ratio_instr_to_constraint": (
                beta_instruction / model_cfg["constraint_beta"]
                if beta_instruction is not None and model_cfg[
                    "constraint_beta"] > 0
                else None
            ),
            "n_empty": fidelity["n_empty"],
            "warnings": fidelity["warnings"],
        }

        model_results[model_key] = result

        # Save checkpoint after each model
        checkpoint_data = {
            "config": {
                "seed_source": SEED_PATH,
                "n_seeds_per_cap": N_SEEDS_PER_CAP,
                "capabilities": CAPABILITIES,
                "n_generations": N_GENERATIONS,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "embedding_method": "TF-IDF cosine",
            },
            "model_results": model_results,
            "completed_models": list(model_results.keys()),
        }
        save_checkpoint(checkpoint_data)
        print(f"  Checkpoint saved: {len(model_results)}/{len(MODELS)} models")

        # Print summary for this model
        print(f"\n  ── {model_key} Summary ──")
        print(f"  β_topic = {beta_topic:.6f} (R²={r2_topic:.4f})" if beta_topic
              else "  β_topic = FAILED")
        print(f"  β_format = {beta_format:.6f} (R²={r2_format:.4f})" if beta_format
              else "  β_format = FAILED")
        print(f"  β_instr = {beta_instruction:.6f} (R²={r2_instr:.4f})" if beta_instruction
              else "  β_instr = FAILED")
        print(f"  β_constraint = {model_cfg['constraint_beta']:.6f}")
        if beta_instruction:
            print(f"  Ratio instr/constraint = "
                  f"{beta_instruction / model_cfg['constraint_beta']:.3f}")
        print(f"  T_k: {{{', '.join(f'G{g}: {fidelity['agg_topic'].get(g, 0):.4f}' for g in gens)}}}")
        print(f"  F_k: {{{', '.join(f'G{g}: {fidelity['agg_format'].get(g, 0):.4f}' for g in gens)}}}")
        print(f"  I_k: {{{', '.join(f'G{g}: {fidelity['agg_composite'].get(g, 0):.4f}' for g in gens)}}}")

    # ── Cross-model comparison ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 70}")

    comparison_rows = []
    for model_cfg in MODELS:
        model_key = model_cfg["name"]
        if model_key not in model_results:
            continue
        r = model_results[model_key]
        comparison_rows.append({
            "model": model_key,
            "beta_constraint": r["constraint_beta"],
            "beta_topic": r["beta_topic"],
            "beta_format": r["beta_format"],
            "beta_instruction": r["beta_instruction"],
            "ratio_instr_to_constraint": r["ratio_instr_to_constraint"],
            "n_seeds_with_beta": r["beta_instruction_bootstrap"].get(
                "n_seeds_with_beta", 0),
        })

    # Print comparison table
    header = (f"{'Model':<24} {'β_constr':>10} {'β_topic':>10} "
              f"{'β_format':>10} {'β_instr':>10} {'Ratio':>8}")
    print(f"\n{header}")
    print("-" * 74)
    for row in comparison_rows:
        bc = row["beta_constraint"]
        bt = row["beta_topic"] or 0.0
        bf = row["beta_format"] or 0.0
        bi = row["beta_instruction"] or 0.0
        ratio = row["ratio_instr_to_constraint"] or 0.0
        print(f"{row['model']:<24} {bc:>10.6f} {bt:>10.6f} "
              f"{bf:>10.6f} {bi:>10.6f} {ratio:>8.3f}")

    # H1: Test beta_instruction != 0
    print(f"\n── H1: β_instruction distinguishable from zero ──")
    for row in comparison_rows:
        r = model_results[row["model"]]
        bs = r["beta_instruction_bootstrap"]
        if bs.get("ci_lower") is not None:
            sig = "YES" if bs["ci_lower"] > 0 else "NO"
            print(f"  {row['model']}: β_instr 95% CI "
                  f"[{bs['ci_lower']:.6f}, {bs['ci_upper']:.6f}], "
                  f"nonzero={sig}")

    # H2: rho(beta_instr, beta_constraint)
    print(f"\n── H2: Correlation β_instruction vs β_constraint ──")
    bc_vals = [r["beta_constraint"] for r in comparison_rows if r["beta_instruction"]]
    bi_vals = [r["beta_instruction"] for r in comparison_rows if r["beta_instruction"]]
    if len(bc_vals) >= 3:
        # Spearman with small n
        from scipy.stats import spearmanr
        rho, pval = spearmanr(bc_vals, bi_vals)
        print(f"  Spearman ρ = {rho:.4f} (p = {pval:.4f}, n = {len(bc_vals)})")
        if abs(rho) < 0.5:
            print("  Result: β_instruction is NOT redundant with β_constraint "
                  "(ρ < 0.5)")
        else:
            print("  Result: β_instruction partially tracks β_constraint "
                  f"(ρ = {rho:.4f})")
    else:
        print(f"  Insufficient models for correlation (need 3, got {len(bc_vals)})")

    # H3: beta_topic vs beta_format within each model
    print(f"\n── H3: Topic vs format decay rates ──")
    for row in comparison_rows:
        bt = row["beta_topic"]
        bf = row["beta_format"]
        if bt is not None and bf is not None:
            if bf > bt:
                print(f"  {row['model']}: format decays FASTER (β_format={bf:.6f} "
                      f"> β_topic={bt:.6f})")
            else:
                print(f"  {row['model']}: topic decays FASTER (β_topic={bt:.6f} "
                      f"> β_format={bf:.6f})")

    # ── Save final output ───────────────────────────────────────────────────
    final_output = {
        "experiment": "instruction_following_control",
        "baseline_number": 7,
        "date": "2026-05-17",
        "description": ("Disentangles topic/format instruction decay from "
                        "constraint-structure degradation under recursion"),
        "method": {
            "topic_fidelity": "TF-IDF cosine similarity vs Gen0 prompt",
            "format_fidelity": "regex/rule-based binary rubrics per capability",
            "composite": "I_k = T_k * F_k",
            "decay_model": "I_k = I_0 * exp(-β * k), OLS on log values",
        },
        "config": {
            "models": [m["name"] for m in MODELS],
            "n_seeds": N_SEEDS_PER_CAP * len(CAPABILITIES),
            "n_capabilities": len(CAPABILITIES),
            "capabilities": CAPABILITIES,
            "n_generations": N_GENERATIONS,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "embedding_method": "TF-IDF cosine",
            "seed_source": "gpt-4o-mini_s100_lineage.jsonl (Gen0 only)",
        },
        "model_results": model_results,
        "comparison": {
            "rows": comparison_rows,
            "h1_beta_instr_nonzero": {
                m["model"]: bool(
                    model_results[m["model"]]["beta_instruction_bootstrap"].get(
                        "ci_lower", 0) > 0
                )
                for m in comparison_rows
            },
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved final results to: {OUTPUT_PATH}")

    print(f"\n{'=' * 70}")
    print("INSTRUCTION-FOLLOWING CONTROL EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")

    return final_output


if __name__ == "__main__":
    main()
