"""β calibration: map β to real downstream quality loss across models × tasks.

Tests 2-3 models on 2 tasks (code + factual), 4 recursive generations.
Builds β → quality_loss calibration curve for interpretable β reporting.
"""
import sys, os, json, time, re, subprocess, tempfile
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

GENERATIONS = 4

# ── Models under test ──
MODELS = [
    ("gpt-4o-mini", "quickrouter", "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"),
    ("deepseek-chat", "deepseek", None),  # uses hardcoded key
]

# ── Code problems ──
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

# ── Factual problems (number/name preservation — degradation = wrong facts) ──
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


def score_code(text: str, test: str) -> float:
    """Compile-and-run test. Returns 0-1."""
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
            return 1.0 if result.returncode == 0 else (0.4 if 'AssertionError' in (result.stderr + result.stdout) else 0.0)
    except Exception:
        return 0.0


def score_fact(text: str, answers: list[str]) -> float:
    """Check if any expected answer appears. Returns 0-1."""
    lower = text.lower()
    return 1.0 if any(a.lower() in lower for a in answers) else 0.0


def run_model_experiment(model: str, provider: str, api_key: str | None, task_type: str,
                         problems: list, score_fn) -> dict:
    """Run recursive generation experiment for one model on one task."""
    adapter = create_provider(provider, model=model, api_key=api_key) if api_key else create_provider(provider, model=model)
    extractor = HybridConstraintExtractor(judge_fn=None)

    gen_scores = {g: [] for g in range(1, GENERATIONS + 1)}
    gen_texts = {g: [] for g in range(1, GENERATIONS + 1)}

    for prob in problems:
        if task_type == "code":
            name, prompt, test = prob
            full_prompt = f"{prompt}\nReturn ONLY the function, no explanation."
        else:
            name, prompt, answers = prob
            full_prompt = prompt

        for gen in range(1, GENERATIONS + 1):
            resp = adapter.generate(full_prompt, max_tokens=200 if task_type == "code" else 80, temperature=0.7)
            time.sleep(getattr(adapter, '_sleep_after', 0.1) if hasattr(adapter, '_sleep_after') else 0.15)

            if task_type == "code":
                score = score_code(resp, test)
            else:
                score = score_fact(resp, answers)

            gen_scores[gen].append(score)
            gen_texts[gen].append(resp)
            full_prompt = resp  # Recursive

    # Quality trajectory
    means = [float(np.mean(gen_scores[g])) for g in range(1, GENERATIONS + 1)]

    # Compute β
    samples = []
    for gen in range(1, GENERATIONS + 1):
        for i in range(len(problems)):
            samples.append(DataSample(
                text=gen_texts[gen][i],
                generation=gen,
                source_model=model,
                capability_tags=[task_type],
                sample_id=f"G{gen}_{i:04d}",
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

    # Linear fit: quality vs generation
    gens_arr = np.arange(1, GENERATIONS + 1)
    slope, _ = np.polyfit(gens_arr, means, 1)
    quality_loss_per_gen = -slope  # positive = loss

    return {
        "model": model, "task": task_type,
        "beta": beta,
        "quality_per_gen": [float(m) for m in means],
        "quality_loss_per_gen": float(quality_loss_per_gen),
        "n_problems": len(problems),
    }


def run():
    print("=" * 60)
    print("β Calibration: multi-model × multi-task")
    print(f"Models: {len(MODELS)} × Tasks: 2 × Gens: {GENERATIONS}")
    print("=" * 60)

    all_results = []

    for model, provider, api_key in MODELS:
        print(f"\n{'─'*40}")
        print(f"Model: {model}")

        # Code task
        print(f"  Code task ({len(CODE_PROBLEMS)} problems)...")
        code_res = run_model_experiment(model, provider, api_key, "code", CODE_PROBLEMS, score_code)
        print(f"    β={code_res['beta']:.4f}, quality_loss/gen={code_res['quality_loss_per_gen']:.4f}")
        print(f"    Quality: {[f'{x:.3f}' for x in code_res['quality_per_gen']]}")
        all_results.append(code_res)

        # Fact task
        print(f"  Fact task ({len(FACT_PROBLEMS)} problems)...")
        fact_res = run_model_experiment(model, provider, api_key, "fact", FACT_PROBLEMS, score_fact)
        print(f"    β={fact_res['beta']:.4f}, quality_loss/gen={fact_res['quality_loss_per_gen']:.4f}")
        print(f"    Quality: {[f'{x:.3f}' for x in fact_res['quality_per_gen']]}")
        all_results.append(fact_res)

    # ── Calibration analysis ──
    print(f"\n{'='*60}")
    print("Calibration Curve: β → Quality Loss per Generation")
    print(f"{'='*60}")

    betas = np.array([r["beta"] for r in all_results])
    losses = np.array([max(r["quality_loss_per_gen"], 0.001) for r in all_results])

    for r in all_results:
        print(f"  {r['model']:20s} {r['task']:6s}  β={r['beta']:.4f}  loss/gen={r['quality_loss_per_gen']:.4f}")

    # Linear regression: loss ~ β
    if len(betas) >= 3:
        slope, intercept = np.polyfit(betas, losses, 1)
        r2 = 1 - np.sum((losses - (intercept + slope * betas))**2) / np.sum((losses - losses.mean())**2)
        pearson = np.corrcoef(betas, losses)[0, 1]

        print(f"\n  Calibration: quality_loss = {slope:.4f} × β + {intercept:.4f}")
        print(f"  R² = {r2:.4f}, Pearson r = {pearson:.4f}")

        if pearson > 0.8:
            print("\n  STRONG: β cleanly maps to quality loss → interpretable β reporting viable")
        elif pearson > 0.5:
            print("\n  MODERATE: β directionally predicts quality loss, calibration improves interpretability")
        else:
            print("\n  WEAK: β and quality loss are decoupled in these experiments")
    else:
        slope, intercept, r2, pearson = 0, 0, 0, 0

    output = {
        "models_tested": len(MODELS),
        "tasks": ["code", "fact"],
        "generations": GENERATIONS,
        "results": all_results,
        "calibration": {
            "slope": float(slope), "intercept": float(intercept),
            "r_squared": float(r2), "pearson_r": float(pearson),
        },
    }
    with open("experiment_data/beta_calibration.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: experiment_data/beta_calibration.json")
    return output


if __name__ == "__main__":
    run()
