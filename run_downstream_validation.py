#!/usr/bin/env python3
"""Downstream task validation: expand model-task pairs for StabilityBench beta calibration.

Matches the data format of synthetic_decay_monitor/beta_calibration.py exactly.
Supports appending new results to experiment_data/beta_calibration.json with
automatic calibration recomputation.

Usage (single model):
  python run_downstream_validation.py --model claude-sonnet-4-6 --provider quickrouter --tasks code,fact --delay 3.0

Usage (batch via config):
  python run_downstream_validation.py --config run_config.json
"""
import sys, os, json, time, re, subprocess, tempfile, argparse, warnings
import numpy as np

# Suppress numpy polyfit deprecation for compatibility with existing code
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load .env ──────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

GENERATIONS = 4  # Must match existing data format

# ── Problem sets (identical to beta_calibration.py for code + fact) ──

CODE_PROBLEMS = [
    ("fib", "Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed).",
     "assert fib(0)==0; assert fib(5)==5; assert fib(10)==55"),
    ("palindrome", "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
     "assert is_palindrome('racecar')==True; assert is_palindrome('hello')==False"),
    ("gcd", "Write a Python function `gcd(a,b)` using Euclidean algorithm.",
     "assert gcd(12,8)==4; assert gcd(17,5)==1"),
    ("merge", "Write a Python function `merge_sorted(a,b)` that merges two sorted lists.",
     "assert merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6]"),
    ("binary_search", "Write a Python function `binary_search(arr,target)` returning index or -1.",
     "assert binary_search([1,2,3,4,5],3)==2; assert binary_search([1,2,3],5)==-1"),
    ("dedup", "Write a Python function `remove_duplicates(lst)` preserving order.",
     "assert remove_duplicates([1,2,2,3,1])==[1,2,3]"),
    ("flatten", "Write a Python function `flatten(nested)` that flattens one level.",
     "assert flatten([[1,2],[3,4]])==[1,2,3,4]"),
    ("anagram", "Write a Python function `anagram(s1,s2)` that checks if strings are anagrams.",
     "assert anagram('listen','silent')==True; assert anagram('hello','world')==False"),
    ("rotate", "Write a Python function `rotate_list(lst,k)` rotating right by k places.",
     "assert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]"),
    ("prime", "Write a Python function `is_prime(n)` returning True if prime.",
     "assert is_prime(2)==True; assert is_prime(4)==False"),
    ("reverse_words", "Write a Python function `reverse_words(s)` reversing each word.",
     "assert reverse_words('hello world')=='olleh dlrow'"),
    ("missing", "Write a Python function `find_missing(arr)` finding missing 0..n.",
     "assert find_missing([0,1,3])==2"),
    ("prefix", "Write a Python function `longest_common_prefix(strs)`.",
     "assert longest_common_prefix(['flower','flow','flight'])=='fl'"),
    ("twosum", "Write a Python function `two_sum(nums,target)` returning indices.",
     "assert two_sum([2,7,11,15],9)==[0,1]"),
    ("parens", "Write a Python function `valid_parentheses(s)` checking balanced brackets.",
     "assert valid_parentheses('()[]{}')==True; assert valid_parentheses('([)]')==False"),
]

FACT_PROBLEMS = [
    ("capital_france", "What is the capital of France? Answer concisely.", ["Paris", "paris"]),
    ("earth_radius", "What is the approximate radius of Earth in kilometers? Answer concisely.", ["6371", "6400"]),
    ("water_boil", "At what temperature in Celsius does water boil at sea level? Answer concisely.", ["100", "212"]),
    ("moons_mars", "How many moons does Mars have? Answer concisely.", ["2", "two"]),
    ("speed_light", "What is the speed of light in km/s approximately? Answer concisely.", ["300000", "299792", "3e5"]),
    ("human_chromosomes", "How many chromosomes do humans have (total, not pairs)? Answer concisely.", ["46"]),
    ("au_distance", "What is 1 AU (astronomical unit) in millions of km approximately? Answer concisely.", ["150", "149.6"]),
    ("element_fe", "What element has the chemical symbol Fe? Answer concisely.", ["Iron", "iron"]),
    ("planet_order", "What is the 4th planet from the Sun? Answer concisely.", ["Mars", "mars"]),
    ("pi_digits", "What is the value of pi to 2 decimal places? Answer concisely.", ["3.14"]),
    ("ocean_deepest", "What is the deepest ocean on Earth? Answer concisely.", ["Pacific", "pacific"]),
    ("largest_organ", "What is the largest organ in the human body? Answer concisely.", ["Skin", "skin"]),
    ("python_year", "In what year was the Python programming language first released? Answer concisely.", ["1991"]),
    ("he_noble", "Is helium a noble gas? Answer yes or no.", ["Yes", "yes"]),
    ("newton_law3", "Newton's third law states: for every action there is an equal and opposite ____? Answer concisely.", ["reaction"]),
]

MATH_PROBLEMS = [
    ("apples_cost", "A store sells apples for $3 each. How much do 7 apples cost? Answer only with the number.", 21),
    ("rectangle_area", "A rectangle has length 8 and width 5. What is its area? Answer only with the number.", 40),
    ("percentage", "What is 15% of 200? Answer only with the number.", 30),
    ("average", "What is the average of 12, 18, and 24? Answer only with the number.", 18),
    ("fraction_decimal", "What is 3/4 expressed as a decimal? Answer only with the number.", 0.75),
    ("prime_check", "Is 17 a prime number? Answer yes or no.", "yes"),
    ("sqrt", "What is the square root of 144? Answer only with the number.", 12),
    ("exponent", "What is 2 to the power of 10? Answer only with the number.", 1024),
    ("solve_equation", "Solve: 2x + 5 = 13. What is x? Answer only with the number.", 4),
    ("coin_probability", "If you flip a fair coin 3 times, what is the probability of getting exactly 3 heads? Answer only with the decimal.", 0.125),
    ("speed_kmh", "A car travels 240 km in 3 hours. What is its average speed in km/h? Answer only with the number.", 80),
    ("discount_price", "A $80 item is on sale with 25% off. What is the sale price in dollars? Answer only with the number.", 60),
    ("simple_interest", "If you invest $1000 at 5% simple interest per year for 3 years, how much interest do you earn in dollars? Answer only with the number.", 150),
    ("ratio_dogs", "If the ratio of cats to dogs is 3:4 and there are 12 cats, how many dogs are there? Answer only with the number.", 16),
    ("hypotenuse", "A right triangle has legs of length 3 and 4. What is the length of the hypotenuse? Answer only with the number.", 5),
]

PROBLEM_SETS = {
    "code": (CODE_PROBLEMS, "code"),
    "fact": (FACT_PROBLEMS, "fact"),
    "math": (MATH_PROBLEMS, "math"),
}


# ── Scoring functions ───────────────────────────────────────────

def score_code(text: str, test: str) -> float:
    """Compile-and-run test. Returns 0-1. Same as beta_calibration.py."""
    match = re.search(r'def\s+\w+[^:]*:.*?(?=\n\S|\n\s*$|$)', text, re.DOTALL)
    if not match:
        return 0.0
    code = match.group(0)
    script = code + "\n\n" + "\n".join(s.strip() for s in test.split(';') if s.strip())
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(script)
            f.flush()
            result = subprocess.run(['python3', f.name], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 1.0
            elif 'AssertionError' in (result.stderr + result.stdout):
                return 0.4
            else:
                return 0.0
    except Exception:
        return 0.0


def score_fact(text: str, answers: list) -> float:
    """Check if any expected answer appears. Returns 0-1. Same as beta_calibration.py."""
    lower = text.lower()
    return 1.0 if any(a.lower() in lower for a in answers) else 0.0


def score_math(text: str, expected) -> float:
    """Score math answer by extracting numbers and comparing to expected value.

    For string answers (yes/no), does substring match.
    For numeric answers, extracts all numbers and checks for equality within tolerance.
    """
    if isinstance(expected, str):
        return 1.0 if expected.lower() in text.lower() else 0.0
    numbers = re.findall(r'-?\d+\.?\d*', text)
    for n in numbers:
        try:
            if abs(float(n) - expected) < 1e-6:
                return 1.0
        except (ValueError, OverflowError):
            pass
    return 0.0


def get_scoring_fn(task_type: str):
    """Return the appropriate scoring function for a task type."""
    if task_type == "code":
        return score_code, "test"
    elif task_type == "fact":
        return score_fact, "answers"
    elif task_type == "math":
        return score_math, "expected"
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_prompt(prob, task_type: str) -> str:
    """Build the initial prompt for a problem based on task type."""
    if task_type == "code":
        name, prompt, test = prob
        return f"{prompt}\nReturn ONLY the function, no explanation."
    elif task_type == "fact":
        name, prompt, answers = prob
        return prompt
    elif task_type == "math":
        name, prompt, expected = prob
        return prompt
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ── Core experiment runner ──────────────────────────────────────

def run_model_experiment(model: str, provider: str, task_type: str,
                         problems: list, score_fn, delay: float,
                         max_tokens: int = 200, temperature: float = 0.7,
                         grader_model: str = None, grader_provider: str = None) -> dict:
    """Run recursive generation experiment for one model on one task.

    Args:
        model: Model name (e.g., "claude-sonnet-4-6")
        provider: Provider name (e.g., "quickrouter")
        task_type: "code", "fact", or "math"
        problems: List of problem tuples
        score_fn: Scoring function (text, extra_arg) -> float
        delay: Seconds to sleep between API calls
        max_tokens: Max tokens for generation
        temperature: Temperature for generation
        grader_model: If set, use this model as LLM judge (overrides score_fn)
        grader_provider: Provider for the grader model

    Returns:
        dict with keys: model, task, beta, quality_per_gen, quality_loss_per_gen, n_problems
    """
    adapter = create_provider(provider, model=model)
    extractor = HybridConstraintExtractor(judge_fn=None)

    gen_scores = {g: [] for g in range(1, GENERATIONS + 1)}
    gen_texts = {g: [] for g in range(1, GENERATIONS + 1)}

    print(f"  Running {model} on {task_type} ({len(problems)} problems, {GENERATIONS} gens)...")

    for pi, prob in enumerate(problems):
        full_prompt = get_prompt(prob, task_type)
        prob_name = prob[0]

        for gen in range(1, GENERATIONS + 1):
            resp = ""
            for attempt in range(3):
                try:
                    resp = adapter.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)
                    if resp:
                        break
                    # Empty response - retry
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                except Exception as e:
                    err_msg = str(e)[:120]
                    if attempt < 2:
                        w = 2 ** (attempt + 1)
                        print(f"    [{prob_name}] gen{gen} retry {attempt+1}/3 in {w}s: {type(e).__name__}: {err_msg}")
                        time.sleep(w)
                    else:
                        print(f"    [{prob_name}] gen{gen} FAIL after 3 attempts: {type(e).__name__}: {err_msg}")
                        resp = "[ERROR]"
            if not resp:
                resp = "[EMPTY]"

            score = score_fn(resp, prob[2])  # prob[2] = test/answers/expected
            gen_scores[gen].append(score)
            gen_texts[gen].append(resp)
            full_prompt = resp  # Recursive: output becomes next input

            time.sleep(delay)

        if (pi + 1) % 5 == 0:
            means_sofar = {g: float(np.mean(gen_scores[g])) for g in range(1, GENERATIONS + 1)}
            print(f"    {pi+1}/{len(problems)} problems done. Gen means: "
                  f"{[f'{means_sofar[g]:.3f}' for g in range(1, GENERATIONS+1)]}")

    # ── Quality trajectory ──
    means = [float(np.mean(gen_scores[g])) for g in range(1, GENERATIONS + 1)]

    # ── Compute beta via DecayEngine ──
    samples = []
    for gen in range(1, GENERATIONS + 1):
        for i in range(len(problems)):
            samples.append(DataSample(
                text=gen_texts[gen][i],
                generation=gen,
                source_model=model,
                capability_tags=[task_type],
                sample_id=f"{task_type}_G{gen}_{i:04d}",
            ))
    lineage = DatasetLineage(samples=samples)
    engine = DecayEngine(lineage, extractor)
    engine.run_all_capabilities()

    betas = []
    for t in engine.get_all_trajectories():
        if "trajectory" in t:
            for g in t["trajectory"]:
                if "beta" in g and g["beta"] > 0:
                    betas.append(g["beta"])
    beta = float(np.mean(betas)) if betas else 0.25

    # ── Linear fit: quality vs generation ──
    gens_arr = np.arange(1, GENERATIONS + 1)
    slope, _ = np.polyfit(gens_arr, means, 1)
    quality_loss_per_gen = -slope  # positive = loss

    result = {
        "model": model,
        "task": task_type,
        "beta": float(beta),
        "quality_per_gen": [float(m) for m in means],
        "quality_loss_per_gen": float(quality_loss_per_gen),
        "n_problems": len(problems),
    }

    print(f"    beta={beta:.4f}  quality_loss/gen={quality_loss_per_gen:.4f}")
    print(f"    Quality per gen: {[f'{m:.3f}' for m in means]}")
    return result


# ── Pearson r p-value via Fisher z-transformation ──────────────

def _pearson_p_value(r, n):
    """Two-tailed p-value for Pearson correlation coefficient.

    Uses Fisher z-transformation:
        z = 0.5 * ln((1+r)/(1-r))  approx  N(0, 1/(n-3))

    Test statistic: z * sqrt(n-3) approx N(0,1) under H0: rho = 0
    p = 2 * (1 - Phi(|z* sqrt(n-3)|))
    """
    import math
    if n <= 3:
        return 1.0
    if abs(r) >= 1.0 - 1e-15:
        return 0.0
    # Fisher z-transformation
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    z_stat = z * math.sqrt(n - 3)
    # Two-tailed p-value from normal distribution
    # Phi(x) = 0.5 * erfc(-x/sqrt(2))
    # tail = 1 - Phi(|z_stat|) = 0.5 * erfc(|z_stat|/sqrt(2))
    tail = 0.5 * math.erfc(abs(z_stat) / math.sqrt(2.0))
    p = 2.0 * tail
    return max(0.0, min(1.0, p))


# ── Calibration recomputation ───────────────────────────────────

def recompute_calibration(results: list) -> dict:
    """Recompute calibration statistics from all results."""
    betas = np.array([r["beta"] for r in results])
    losses = np.array([max(r["quality_loss_per_gen"], 0.001) for r in results])

    if len(betas) >= 3:
        slope, intercept = np.polyfit(betas, losses, 1)
        ss_res = np.sum((losses - (intercept + slope * betas))**2)
        ss_tot = np.sum((losses - losses.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        pearson = float(np.corrcoef(betas, losses)[0, 1])
    else:
        slope, intercept, r2, pearson = 0.0, 0.0, 0.0, 0.0

    # Compute p-value for Pearson r (two-tailed, Fisher z-transformation)
    if len(betas) >= 4:
        p_value = _pearson_p_value(pearson, len(betas))
    else:
        p_value = 1.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "pearson_r": float(pearson),
        "n_pairs": len(results),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


# ── Main ────────────────────────────────────────────────────────

def load_calibration(path: str) -> dict:
    """Load existing calibration JSON or return empty template."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "models_tested": 0,
        "tasks": [],
        "generations": GENERATIONS,
        "results": [],
        "calibration": {},
    }


def save_calibration(path: str, data: dict):
    """Save calibration data with atomic write."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description="Run downstream task validation for StabilityBench beta calibration"
    )
    parser.add_argument("--model", help="Model name (e.g., claude-sonnet-4-6)")
    parser.add_argument("--provider", help="Provider name (e.g., quickrouter, deepseek)")
    parser.add_argument("--tasks", default="code,fact",
                       help="Comma-separated task types (code,fact,math)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--output", default="experiment_data/beta_calibration.json",
                       help="Path to calibration JSON file")
    parser.add_argument("--config", help="JSON config file for batch runs (alternative to --model)")
    parser.add_argument("--list-completed", action="store_true",
                       help="List completed model-task pairs and exit")
    parser.add_argument("--force", action="store_true",
                       help="Re-run even if model-task pair already exists")
    args = parser.parse_args()

    # ── Determine output path ──
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load existing data ──
    data = load_calibration(output_path)
    existing = {(r["model"], r["task"]) for r in data.get("results", [])}

    if args.list_completed:
        print("Completed model-task pairs:")
        for model, task in sorted(existing):
            print(f"  {model:30s} {task}")
        print(f"\nTotal: {len(existing)} pairs")
        return

    # ── Build run list ──
    runs = []
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for entry in cfg.get("models", []):
            runs.append({
                "model": entry["model"],
                "provider": entry["provider"],
                "tasks": entry.get("tasks", "code,fact"),
                "delay": entry.get("delay", 1.0),
                "max_tokens": entry.get("max_tokens", 200),
                "temperature": entry.get("temperature", 0.7),
            })
    elif args.model:
        runs.append({
            "model": args.model,
            "provider": args.provider,
            "tasks": args.tasks,
            "delay": args.delay,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        })
    else:
        parser.error("Either --model/--provider or --config is required")

    # ── Run experiments ──
    all_tasks_seen = set(data.get("tasks", []))
    models_seen = set(r["model"] for r in data.get("results", []))

    for run_i, run in enumerate(runs):
        task_list = [t.strip() for t in run["tasks"].split(",")]
        model = run["model"]
        provider = run["provider"]

        print(f"\n{'='*60}")
        print(f"Run {run_i+1}/{len(runs)}: {model} via {provider}")
        print(f"  Tasks: {task_list}  Delay: {run['delay']}s  Temp: {run['temperature']}")
        print(f"{'='*60}")

        for task_type in task_list:
            if task_type not in PROBLEM_SETS:
                print(f"  SKIP: unknown task type '{task_type}'. Known: {list(PROBLEM_SETS.keys())}")
                continue

            pair_key = (model, task_type)
            if pair_key in existing and not args.force:
                print(f"  SKIP: {model}/{task_type} already exists. Use --force to re-run.")
                continue

            problems, _ = PROBLEM_SETS[task_type]
            score_fn, _ = get_scoring_fn(task_type)
            max_tok = run["max_tokens"]

            # Adjust max_tokens for task types
            if task_type == "fact":
                max_tok = min(max_tok, 80)
            elif task_type == "math":
                max_tok = min(max_tok, 100)

            try:
                result = run_model_experiment(
                    model=model,
                    provider=provider,
                    task_type=task_type,
                    problems=problems,
                    score_fn=score_fn,
                    delay=run["delay"],
                    max_tokens=max_tok,
                    temperature=run["temperature"],
                )
                data["results"].append(result)
                existing.add(pair_key)
                models_seen.add(model)
                all_tasks_seen.add(task_type)

                # ── Save intermediate results after each task ──
                data["models_tested"] = len(models_seen)
                data["tasks"] = sorted(all_tasks_seen)
                data["calibration"] = recompute_calibration(data["results"])
                save_calibration(output_path, data)
                print(f"  Saved intermediate results ({len(data['results'])} pairs total)")

            except Exception as e:
                print(f"  ERROR on {model}/{task_type}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # Save what we have so far
                data["models_tested"] = len(models_seen)
                data["tasks"] = sorted(all_tasks_seen)
                data["calibration"] = recompute_calibration(data["results"])
                save_calibration(output_path, data)
                print(f"  Saved partial results after error ({len(data['results'])} pairs)")

    # ── Final calibration ──
    data["models_tested"] = len(models_seen)
    data["tasks"] = sorted(all_tasks_seen)
    data["generations"] = GENERATIONS
    calibration = recompute_calibration(data["results"])
    data["calibration"] = calibration
    save_calibration(output_path, data)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(data['results'])} model-task pairs")
    print(f"  Models: {data['models_tested']}  Tasks: {data['tasks']}")
    print(f"{'='*60}")
    for r in data["results"]:
        print(f"  {r['model']:30s} {r['task']:6s}  beta={r['beta']:.4f}  "
              f"loss/gen={r['quality_loss_per_gen']:.4f}  Q={[f'{q:.3f}' for q in r['quality_per_gen']]}")

    print(f"\n  Calibration: quality_loss = {calibration['slope']:.4f} x beta + {calibration['intercept']:.4f}")
    print(f"  R^2 = {calibration['r_squared']:.4f}")
    print(f"  Pearson r = {calibration['pearson_r']:.4f}")
    print(f"  p-value = {calibration['p_value']:.4f}")
    print(f"  n = {calibration['n_pairs']} model-task pairs")
    if calibration.get("significant_at_0.05"):
        print(f"  STATISTICALLY SIGNIFICANT at p < 0.05!")
    else:
        print(f"  NOT significant (need p < 0.05). n_pairs={calibration['n_pairs']}")
        if calibration['n_pairs'] < 12:
            print(f"  Suggest adding {12 - calibration['n_pairs']} more pairs to reach n >= 12")

    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
