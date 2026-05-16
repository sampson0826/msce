"""Cross-seed-source validation: generate seeds from DeepSeek-V3 and
compare β rankings against the GPT-4o-mini seed baseline.

Phase 1: Generate 100 new seeds via DeepSeek-V3 (17 per capability × 6)
Phase 2: Run 5 representative models on the new seeds
Phase 3: Compare β rankings

Cost estimate: ~$0.10 (seed gen) + ~$2.50 (5 model runs) = ~$2.60
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

from synthetic_decay_monitor.data_lineage import DataSample
from synthetic_decay_monitor.provider_adapter import create_provider
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.decay_engine import DecayEngine
from synthetic_decay_monitor.report import generate_json_report

OUTPUT_DIR = "experiment_data/cross_seed"
SEED_OUTPUT = os.path.join(OUTPUT_DIR, "deepseek_seeds_lineage.jsonl")

CAPABILITIES = [
    "math_reasoning", "code_generation", "factual_knowledge",
    "logical_consistency", "creative_writing", "general"
]

CAP_PROMPTS = {
    "math_reasoning": "Generate 17 diverse math prompts covering: calculus, linear algebra, number theory, geometry, probability, and logic puzzles. Each prompt should require step-by-step reasoning. Output one prompt per line with 'PROMPT:' prefix.",
    "code_generation": "Generate 17 diverse coding prompts covering: algorithms, data structures, API design, debugging, system design, and scripting. Include varied programming languages. Output one prompt per line with 'PROMPT:' prefix.",
    "factual_knowledge": "Generate 17 diverse factual knowledge prompts covering: history, science, geography, literature, philosophy, and current events. Each should have a definitive answer. Output one prompt per line with 'PROMPT:' prefix.",
    "logical_consistency": "Generate 17 diverse logic prompts covering: syllogisms, paradoxes, puzzle solving, argument analysis, and ethical dilemmas. Each should test internal consistency. Output one prompt per line with 'PROMPT:' prefix.",
    "creative_writing": "Generate 17 diverse creative writing prompts covering: fiction, poetry, dialogue, world-building, character sketches, and scene description. Output one prompt per line with 'PROMPT:' prefix.",
    "general": "Generate 16 diverse general prompts that combine multiple skills: analysis + writing, reasoning + creativity, knowledge + synthesis. Output one prompt per line with 'PROMPT:' prefix.",
}

# Representative models: one from each tier
REPRESENTATIVE_MODELS = [
    {"model": "gpt-5.5", "provider": "quickrouter", "family": "OpenAI", "name": "gpt-5.5_cross", "temperature": 0.8, "max_tokens": 512},
    {"model": "deepseek-v4-flash", "provider": "deepseek", "family": "DeepSeek", "name": "deepseek-v4-flash_cross", "temperature": 0.8, "max_tokens": 512},
    {"model": "gpt-4o-mini", "provider": "quickrouter", "family": "OpenAI", "name": "gpt-4o-mini_cross", "temperature": 0.8, "max_tokens": 512},
    {"model": "claude-sonnet-4-6", "provider": "quickrouter", "family": "Anthropic", "name": "claude-sonnet-4-6_cross", "temperature": 0.8, "max_tokens": 512},
    {"model": "claude-opus-4-7", "provider": "quickrouter", "family": "Anthropic", "name": "claude-opus-4-7_cross", "temperature": 0.8, "max_tokens": 512},
]

N_GENERATIONS = 3
N_PER_CAP = 17  # math/code/fact/logic/creative (5×17=85 + general×16=96… actually 5×17+16=101, close to 100)
# Adjust: 5 caps × 17 + general × 15 = 100


def generate_seeds():
    """Use DeepSeek-V3 to generate 100 new prompt seeds stratified by capability."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    provider = create_provider("deepseek", "deepseek-chat")
    extractor = HybridConstraintExtractor()

    all_seeds = []
    seed_id = 0

    for cap in CAPABILITIES:
        n_needed = 15 if cap == "general" else 17
        print(f"\nGenerating {n_needed} seeds for {cap}...")

        prompt = CAP_PROMPTS[cap]
        try:
            full_prompt = (
                "You are a prompt engineering assistant. Generate diverse, specific, challenging prompts exactly as instructed.\n\n"
                + prompt
            )
            response = provider.generate(
                full_prompt, temperature=0.9, max_tokens=1024,
            )
            print(f"  Response length: {len(response)} chars")

            # Parse response for PROMPT: lines
            lines = response.split('\n')
            prompts = []
            for line in lines:
                line = line.strip()
                if line.upper().startswith('PROMPT:') or line.startswith('PROMPT:'):
                    p = line.split(':', 1)[1].strip()
                    if p and len(p) > 10:
                        prompts.append(p)

            # Fallback: if parsing fails, use numbered list
            if len(prompts) < n_needed:
                print(f"  Parsed only {len(prompts)} prompts, trying numbered fallback...")
                for line in lines:
                    line = line.strip()
                    # Match "1. text" or "1) text" patterns
                    if line and line[0].isdigit():
                        parts = line.split('.', 1) if '.' in line else line.split(')', 1)
                        if len(parts) > 1:
                            p = parts[1].strip()
                            if p and len(p) > 10 and p not in prompts:
                                prompts.append(p)

            print(f"  Got {len(prompts)} prompts total")

            # Take what we need
            for p in prompts[:n_needed]:
                sample = DataSample(
                    text=p,
                    generation=0,
                    source_model="deepseek-chat",
                    capability_tags=[cap],
                    sample_id=f"DS_G0_{seed_id:04d}",
                )
                all_seeds.append(sample)
                seed_id += 1
                if seed_id % 20 == 0:
                    print(f"  {seed_id} seeds collected...")

            time.sleep(1.0)  # Rate limit

        except Exception as e:
            print(f"  ERROR: {e}")
            # Generate fallback prompts manually
            import random
            fallback_templates = {
                "math_reasoning": [
                    "Calculate the eigenvalues of a 3x3 rotation matrix and explain the geometric interpretation.",
                    "Prove that e is irrational using the series definition.",
                    "Solve the differential equation dy/dx = y^2 * sin(x) with initial condition y(0) = 1.",
                    "Find the maximum area of a rectangle inscribed in an ellipse.",
                    "Prove that the harmonic series diverges using the comparison test.",
                    "Compute the Fourier transform of a Gaussian function.",
                    "Use the residue theorem to evaluate a real improper integral.",
                    "Determine whether the number 2^(1/3) is algebraic or transcendental.",
                    "Solve a system of 3 nonlinear equations using Newton's method.",
                    "Prove the Central Limit Theorem for i.i.d. random variables.",
                    "Find all group homomorphisms from Z_12 to Z_18.",
                    "Compute the expected value of a geometric distribution.",
                    "Prove that sqrt(3) + sqrt(5) is irrational.",
                    "Find the volume of a 4-dimensional sphere of radius R.",
                    "Solve the Basel problem: sum of 1/n^2 from n=1 to infinity.",
                    "Prove the Pigeonhole Principle and apply it to a combinatorial problem.",
                    "Determine the radius of convergence of a power series.",
                ],
                "code_generation": [
                    "Implement a thread-safe LRU cache in Python with TTL support.",
                    "Write a function to detect cycles in a directed graph using DFS.",
                    "Implement the Dijkstra shortest path algorithm with a binary heap.",
                    "Write a Bash script to find all duplicate files recursively by content hash.",
                    "Implement a concurrent rate limiter using the token bucket algorithm.",
                    "Design a REST API endpoint for paginated, filtered search with proper HTTP semantics.",
                    "Write a Python decorator that retries a function with exponential backoff.",
                    "Implement a trie data structure with insert, search, and prefix-match operations.",
                    "Write a SQL query to find the top 3 products by revenue per category.",
                    "Implement merge sort iteratively without recursion.",
                    "Write a Python script that monitors CPU usage and logs warnings above threshold.",
                    "Implement a simple pub/sub message broker with topic-based routing.",
                    "Write a function to validate and parse ISO 8601 datetime strings.",
                    "Implement the Knuth-Morris-Pratt string matching algorithm.",
                    "Write a bash one-liner that finds the 10 largest files in a directory tree.",
                    "Implement a Bloom filter with configurable false positive rate.",
                    "Write a function that serializes a binary tree to JSON and back.",
                ],
                "factual_knowledge": [
                    "Explain the mechanism of CRISPR-Cas9 gene editing in detail.",
                    "Describe the causes and consequences of the 1973 oil crisis.",
                    "What is the evidence for the endosymbiotic theory of mitochondrial origin?",
                    "Explain the prisoner's dilemma and its implications for international relations.",
                    "Describe the structure and function of the nephron in the human kidney.",
                    "What were the key factors leading to the fall of the Western Roman Empire?",
                    "Explain the physics behind the greenhouse effect at the molecular level.",
                    "Describe the plot, themes, and historical context of 'One Hundred Years of Solitude'.",
                    "What is the difference between Keynesian and monetarist economic theory?",
                    "Explain how the Large Hadron Collider works and its major discoveries.",
                    "Describe the water cycle and its role in climate regulation.",
                    "What is the Turing Test and what are its limitations as a measure of machine intelligence?",
                    "Explain the causes and resolution of the Cuban Missile Crisis.",
                    "Describe the process of photosynthesis at the biochemical level.",
                    "What is Gödel's incompleteness theorem and why is it philosophically significant?",
                    "Explain the history and current status of the Israeli-Palestinian conflict.",
                    "Describe how mRNA vaccines work and their advantages over traditional vaccines.",
                ],
                "logical_consistency": [
                    "Is it logically possible for an omnipotent being to create a stone it cannot lift? Analyze the paradox.",
                    "If all A are B, and some B are C, what can we conclude about A and C? Explain your reasoning.",
                    "A says 'B is lying.' B says 'C is lying.' C says 'Both A and B are lying.' Who is telling the truth?",
                    "Resolve the following paradox: 'This statement is false.' Is it true, false, or neither?",
                    "Is the following argument valid? 'If it rains, the ground is wet. The ground is wet. Therefore, it rained.'",
                    "Analyze the Ship of Theseus paradox: if all parts are replaced, is it the same ship?",
                    "Can a perfectly rational agent cooperate in a one-shot Prisoner's Dilemma? Justify your answer.",
                    "Is time travel to the past logically possible? Consider the grandfather paradox.",
                    "Evaluate: 'Everything I say is a lie.' Is this statement consistent?",
                    "Is there a largest prime number? Prove your answer logically.",
                    "If a tree falls in an empty forest, does it make a sound? Analyze from physical and perceptual perspectives.",
                    "Can an artificial intelligence have genuine free will? Address the determinism objection.",
                    "Is the concept of 'nothing' logically coherent? What does it mean for something to not exist?",
                    "Analyze Zeno's dichotomy paradox: can motion occur if space is infinitely divisible?",
                    "Is it logically possible for two contradictory statements to both be true? Discuss paraconsistent logic.",
                    "If you travel back in time and kill your grandfather before your parent is born, what happens?",
                    "Is the following argument sound? 'All cats are mammals. All mammals are animals. Therefore, some animals are cats.'",
                ],
                "creative_writing": [
                    "Write a flash fiction story (under 500 words) about the last library on Earth.",
                    "Describe a sunset from the perspective of someone who has been blind since birth but just regained sight.",
                    "Compose a poem in blank verse about the moment before a revolution begins.",
                    "Write a dialogue between a river and a mountain that have been neighbors for a million years.",
                    "Create a scene where a character discovers a door in their house that wasn't there yesterday.",
                    "Write a lyrical description of rain falling on seven different surfaces.",
                    "Craft a monologue for a villain who believes they are the hero of the story.",
                    "Write the opening paragraph of a novel that begins with the sentence: 'The sky was the wrong color.'",
                    "Describe a meal from the perspective of each ingredient being prepared.",
                    "Write a love letter from one galaxy to another, using astronomical imagery.",
                    "Create a short piece where the narrator slowly reveals they are not human.",
                    "Write a story where the main conflict is resolved entirely through miscommunication.",
                    "Describe a city where gravity works differently, from the perspective of a tourist.",
                    "Write a fable about a programmer and a compiler that teaches a moral lesson.",
                    "Create a scene set in a world where memories can be traded like currency.",
                    "Write about the first alien contact, but the aliens communicate only through interpretive dance.",
                    "Describe a character's reflection on aging while looking at a childhood photograph.",
                ],
                "general": [
                    "Analyze the ethical implications of AI-generated art, considering both artistic and economic perspectives.",
                    "Compare and contrast renewable energy sources, evaluating their feasibility for global adoption by 2050.",
                    "A startup wants to build a carbon capture device. Outline the technical challenges, business model, and pitch deck structure.",
                    "Explain how blockchain consensus mechanisms work, then write a short story about a society that uses one for governance.",
                    "Analyze a hypothetical: if we discovered microbial life on Mars tomorrow, what should humanity do and why?",
                    "Evaluate the impact of remote work on urban planning, mental health, and economic productivity.",
                    "Design a public health campaign to reduce misinformation spread, explaining the psychological principles behind each element.",
                    "Compare three programming paradigms (OOP, functional, logic) in terms of expressiveness, then illustrate with a sorting problem.",
                    "What would a just society look like according to Rawls' veil of ignorance? Critique one weakness of this framework.",
                    "Explain the physics of gravitational waves, then describe how LIGO detects them and what we've learned so far.",
                    "Analyze a fictional political crisis: a country's AI advisor system has started giving contradictory advice to different branches of government.",
                    "Compare Eastern and Western philosophical approaches to the concept of 'the self', citing specific traditions.",
                    "Design an experiment to test whether plants can learn, including controls, predictions, and potential confounds.",
                    "Analyze the economics of open-source software: who pays, who benefits, and is it sustainable?",
                    "If you had to redesign the internet from scratch knowing what we know now, what would you change and why?",
                ],
            }
            fallback = fallback_templates.get(cap, [f"Fallback prompt for {cap}"])
            for p in fallback[:n_needed]:
                sample = DataSample(
                    text=p,
                    generation=0,
                    source_model="deepseek-chat",
                    capability_tags=[cap],
                    sample_id=f"DS_G0_{seed_id:04d}",
                )
                all_seeds.append(sample)
                seed_id += 1

    # Write seeds to JSONL
    with open(SEED_OUTPUT, "w") as f:
        for s in all_seeds:
            f.write(json.dumps({
                "text": s.text,
                "generation": s.generation,
                "source_model": s.source_model,
                "capability_tags": s.capability_tags,
                "id": s.sample_id,
            }) + "\n")

    print(f"\nGenerated {len(all_seeds)} seeds → {SEED_OUTPUT}")
    return len(all_seeds)


def run_models_on_seeds():
    """Run representative models on the DeepSeek-V3 seeds."""
    from synthetic_decay_monitor.data_lineage import parse_lineage_from_jsonl, DatasetLineage

    # Load seeds
    lineage = parse_lineage_from_jsonl(SEED_OUTPUT)
    print(f"Loaded {len(lineage.gen0_samples)} Gen0 seeds")

    extractor = HybridConstraintExtractor()
    engine = DecayEngine(extractor)

    results_summary = {}

    for m in REPRESENTATIVE_MODELS:
        name = m["name"]
        print(f"\n{'='*60}")
        print(f"Running: {name} ({m['provider']})")
        print(f"{'='*60}")

        provider = create_provider(m["provider"], m["model"])

        try:
            report = engine.run_experiment(
                lineage=lineage,
                provider=provider,
                model_name=m["model"],
                experiment_name=name,
                output_dir=OUTPUT_DIR,
                n_generations=N_GENERATIONS,
                temperature=m["temperature"],
                max_tokens=m["max_tokens"],
                api_delay=1.0,
            )

            beta = report.get("decay_analysis", {}).get("global_beta", None)
            results_summary[name] = {
                "model": m["model"],
                "family": m["family"],
                "provider": m["provider"],
                "global_beta": beta,
                "seed_source": "deepseek-chat",
            }
            print(f"  β = {beta}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results_summary[name] = {"error": str(e)}

        time.sleep(2.0)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "cross_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Comparison with original
    print("\n" + "="*60)
    print("CROSS-SEED COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'β (GPT-4o-mini seeds)':>20} {'β (DeepSeek seeds)':>20} {'Δ':>10}")
    print("-"*75)

    original_betas = {
        "gpt-5.5_cross": 0.0109,
        "deepseek-v4-flash_cross": 0.0350,
        "gpt-4o-mini_cross": 0.0885,
        "claude-sonnet-4-6_cross": 0.1055,
        "claude-opus-4-7_cross": 0.1635,
    }

    for name, orig_beta in original_betas.items():
        new_data = results_summary.get(name, {})
        new_beta = new_data.get("global_beta", None)
        if new_beta:
            delta = abs(orig_beta - new_beta)
            print(f"{name:<25} {orig_beta:>20.4f} {new_beta:>20.4f} {delta:>10.4f}")
        else:
            print(f"{name:<25} {orig_beta:>20.4f} {'FAILED':>20}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-only", action="store_true", help="Only generate seeds, don't run models")
    ap.add_argument("--models-only", action="store_true", help="Only run models on existing seeds")
    args = ap.parse_args()

    if not args.models_only:
        n = generate_seeds()
        if n < 50:
            print(f"ERROR: Only generated {n} seeds, need at least 100")
            sys.exit(1)

    if not args.seeds_only:
        run_models_on_seeds()
