"""β → downstream loss mapping v2: code generation + compilation test.

Uses 20 Python coding problems through 4 recursive generations.
Ground truth: does the generated code execute correctly?
"""
import sys, os, json, time, re, subprocess, tempfile
import numpy as np

sys.path.insert(0, ".")
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.data_lineage import DataSample, DatasetLineage

KEY = "sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA"
MODEL = "gpt-4o-mini"
N_GENERATIONS = 4

CODING_PROBLEMS = [
    ("Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed).\nassert fib(0)==0; assert fib(5)==5; assert fib(10)==55",
     "fib(0)==0;fib(5)==5;fib(10)==55"),
    ("Write a Python function `is_palindrome(s)` that returns True if string s is a palindrome.\nassert is_palindrome('racecar')==True; assert is_palindrome('hello')==False",
     "is_palindrome('racecar')==True;is_palindrome('hello')==False"),
    ("Write a Python function `gcd(a, b)` that returns the greatest common divisor using Euclidean algorithm.\nassert gcd(12,8)==4; assert gcd(17,5)==1; assert gcd(100,10)==10",
     "gcd(12,8)==4;gcd(17,5)==1"),
    ("Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.\nassert merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6]",
     "merge_sorted([1,3,5],[2,4,6])=="),
    ("Write a Python function `binary_search(arr, target)` that returns index or -1.\nassert binary_search([1,2,3,4,5],3)==2; assert binary_search([1,2,3],5)==-1",
     "binary_search([1,2,3,4,5],3)==2"),
    ("Write a Python function `remove_duplicates(lst)` that returns list with duplicates removed preserving order.\nassert remove_duplicates([1,2,2,3,1])==[1,2,3]",
     "remove_duplicates([1,2,2,3,1])=="),
    ("Write a Python function `count_words(text)` that returns dict of word→count.\nassert count_words('a b a')=={'a':2,'b':1}",
     "count_words('a b a')=="),
    ("Write a Python function `flatten(nested)` that flattens a nested list one level deep.\nassert flatten([[1,2],[3,4]])==[1,2,3,4]",
     "flatten([[1,2],[3,4]])=="),
    ("Write a Python function `anagram(s1, s2)` that returns True if strings are anagrams.\nassert anagram('listen','silent')==True; assert anagram('hello','world')==False",
     "anagram('listen','silent')==True"),
    ("Write a Python function `rotate_list(lst, k)` that rotates list right by k positions.\nassert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]",
     "rotate_list([1,2,3,4,5],2)=="),
    ("Write a Python function `is_prime(n)` that returns True if n is prime.\nassert is_prime(2)==True; assert is_prime(4)==False; assert is_prime(17)==True",
     "is_prime(2)==True;is_prime(4)==False"),
    ("Write a Python function `reverse_words(s)` that reverses each word in a sentence.\nassert reverse_words('hello world')=='olleh dlrow'",
     "reverse_words('hello world')=="),
    ("Write a Python function `find_missing(arr)` that finds the missing number in 0..n.\nassert find_missing([0,1,3])==2; assert find_missing([0,1,2,4])==3",
     "find_missing([0,1,3])==2"),
    ("Write a Python function `longest_common_prefix(strs)` that returns the longest common prefix of a list of strings.\nassert longest_common_prefix(['flower','flow','flight'])=='fl'",
     "longest_common_prefix(['flower','flow','flight'])=="),
    ("Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add to target.\nassert two_sum([2,7,11,15],9)==[0,1]",
     "two_sum([2,7,11,15],9)=="),
    ("Write a Python function `valid_parentheses(s)` that returns True if parentheses are balanced.\nassert valid_parentheses('()[]{}')==True; assert valid_parentheses('([)]')==False",
     "valid_parentheses('()[]{}')==True"),
    ("Write a Python function `majority_element(nums)` that returns the element appearing more than n/2 times.\nassert majority_element([3,2,3])==3",
     "majority_element([3,2,3])=="),
    ("Write a Python function `max_subarray(nums)` that returns the maximum subarray sum (Kadane's algorithm).\nassert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6",
     "max_subarray([-2,1,-3,4,-1,2,1,-5,4])=="),
    ("Write a Python function `sqrt_newton(n, eps)` that approximates sqrt using Newton's method.\nassert abs(sqrt_newton(2,0.001)-1.414)<0.01",
     "abs(sqrt_newton(2,0.001)-1.414)<0.01"),
    ("Write a Python function `transpose(matrix)` that transposes a 2D list.\nassert transpose([[1,2],[3,4]])==[[1,3],[2,4]]",
     "transpose([[1,2],[3,4]])=="),
]


def extract_function(code: str) -> str | None:
    """Extract function definition from generated text."""
    # Find def ... block
    match = re.search(r'def\s+\w+[^:]*:.*?(?=\n\S|\n\s*$|$)', code, re.DOTALL)
    if match:
        return match.group(0)
    # Fallback: take everything after first 'def'
    lines = code.split('\n')
    func_lines = []
    in_func = False
    for line in lines:
        if line.startswith('def ') or line.startswith('    def '):
            in_func = True
        if in_func:
            if line and not line[0].isspace() and not line.startswith('def') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines) if func_lines else None


def test_code(code: str, test_assert: str) -> float:
    """Test generated code against assertions. Returns 0-1 score."""
    if not code:
        return 0.0

    # Build test script
    test_script = code + "\n\n# Tests\n"
    for assert_stmt in test_assert.split(';'):
        assert_stmt = assert_stmt.strip()
        if not assert_stmt:
            continue
        # Handle '==' checks (not full assert statements)
        if assert_stmt.startswith('assert '):
            test_script += assert_stmt + "\n"
        elif '==' in assert_stmt and not assert_stmt.startswith('assert'):
            test_script += f"assert {assert_stmt}\n"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(test_script)
            f.flush()
            result = subprocess.run(
                ['python3', f.name], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return 1.0
            # Partial: no syntax error but assertion failed
            if 'AssertionError' in result.stderr or 'AssertionError' in result.stdout:
                return 0.5
            return 0.0
    except Exception:
        return 0.0


def run():
    print("=" * 60)
    print(f"β → Code Quality Mapping: {MODEL}")
    print(f"Problems: {len(CODING_PROBLEMS)}, Generations: {N_GENERATIONS}")
    print("=" * 60)

    adapter = create_provider("quickrouter", model=MODEL, api_key=KEY)
    extractor = HybridConstraintExtractor(judge_fn=None)

    gen_results = {g: [] for g in range(1, N_GENERATIONS + 1)}
    texts_per_gen = {g: [] for g in range(1, N_GENERATIONS + 1)}

    for i, (problem, test) in enumerate(CODING_PROBLEMS):
        prompt = problem

        for gen in range(1, N_GENERATIONS + 1):
            resp = adapter.generate(prompt, max_tokens=256, temperature=0.7)
            time.sleep(0.15)

            func_code = extract_function(resp)
            score = test_code(func_code, test) if func_code else 0.0

            gen_results[gen].append((problem, resp, score))
            texts_per_gen[gen].append(resp)
            prompt = resp  # Feed output as next input

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(CODING_PROBLEMS)}] done")

    # Per-generation scores
    print(f"\n{'='*60}")
    print(f"Code correctness across recursive generations:")
    print(f"{'='*60}")
    gen_scores = []
    for gen in range(1, N_GENERATIONS + 1):
        scores = [s for _, _, s in gen_results[gen]]
        mean_s = np.mean(scores)
        gen_scores.append(mean_s)
        n_pass = sum(1 for s in scores if s >= 0.9)
        n_part = sum(1 for s in scores if 0.4 <= s < 0.9)
        n_fail = sum(1 for s in scores if s < 0.4)
        print(f"  Gen{gen}: score={mean_s:.3f} (pass={n_pass} partial={n_part} fail={n_fail})")

    # Compute β
    samples = []
    for gen in range(1, N_GENERATIONS + 1):
        for i in range(len(CODING_PROBLEMS)):
            samples.append(DataSample(
                text=texts_per_gen[gen][i],
                generation=gen,
                source_model=MODEL,
                capability_tags=["code_generation"],
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
    beta = sum(betas) / len(betas) if betas else 0.25

    print(f"\n{'='*60}")
    print(f"β → Code quality mapping:")
    print(f"{'='*60}")
    print(f"  Measured β: {beta:.4f}")

    # β predicted decay: score_n = score_1 * (1 - β)^(n-1)
    predicted = [gen_scores[0] * (1 - beta)**(n) for n in range(N_GENERATIONS)]

    rel_drop_obs = gen_scores[0] - gen_scores[-1]
    rel_drop_pred = predicted[0] - predicted[-1]

    for gen in range(1, N_GENERATIONS + 1):
        print(f"  Gen{gen}: observed={gen_scores[gen-1]:.3f}, β-predicted={predicted[gen-1]:.3f}")

    # Correlation
    gens_arr = np.arange(1, N_GENERATIONS + 1)
    obs_arr = np.array(gen_scores)
    r = np.corrcoef(gens_arr, obs_arr)[0, 1]

    print(f"\n  Observed quality drop: {rel_drop_obs:.3f}")
    print(f"  β-predicted quality drop: {rel_drop_pred:.3f}")
    print(f"  Pearson r (gen vs score): {r:.4f}")

    # Verdict
    print(f"\n{'='*60}")
    if r < -0.6 and abs(rel_drop_obs - rel_drop_pred) < 0.3:
        print("VERDICT: β predicts downstream code quality degradation")
    elif r < -0.3:
        print("VERDICT: Moderate correlation — β directionally correct")
    else:
        print("VERDICT: Weak correlation — need harder problems or more generations")

    output = {
        "model": MODEL, "n_problems": len(CODING_PROBLEMS),
        "generations": N_GENERATIONS, "beta": beta,
        "per_gen_scores": [float(s) for s in gen_scores],
        "predicted_scores": [float(s) for s in predicted],
        "pearson_r": float(r),
    }
    with open("experiment_data/beta_downstream_code.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: experiment_data/beta_downstream_code.json")
    return output


if __name__ == "__main__":
    run()
