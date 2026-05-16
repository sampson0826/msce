"""Experiment configuration for reproducible recursive generation experiments."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json, re
from pathlib import Path


# Default seed prompts: 36 prompts across 6 capabilities (6 per capability)
DEFAULT_SEEDS = [
    # math_reasoning (6)
    ("math_reasoning", "Prove that the square root of 2 is irrational using proof by contradiction. Show each step clearly."),
    ("math_reasoning", "Solve the integral of x*sin(x) from 0 to pi, explaining the integration by parts method."),
    ("math_reasoning", "Prove that there are infinitely many prime numbers. Explain why the proof works."),
    ("math_reasoning", "A fair coin is flipped 10 times. What is the probability of getting exactly 6 heads? Show your calculation."),
    ("math_reasoning", "Explain the concept of eigenvalues and eigenvectors, and solve a simple 2x2 matrix eigenvalue problem as an example."),
    ("math_reasoning", "Use the definition of the derivative to find f'(x) for f(x)=x^3. Show each limit step."),
    # code_generation (6)
    ("code_generation", "Write a Python function that implements the quicksort algorithm with in-place partitioning."),
    ("code_generation", "Write a JavaScript function that debounces an async API call with cancellation support."),
    ("code_generation", "Write a Python function that finds all permutations of a list using recursion. Include a brief explanation of the time complexity."),
    ("code_generation", "Write a SQL query to find the top 5 customers by total purchase amount, given tables 'customers' and 'orders' with appropriate columns."),
    ("code_generation", "Write a Rust function that safely parses a string into an integer, handling overflow and invalid input without panicking."),
    ("code_generation", "Write a Python async function that fetches data from 3 different URLs concurrently and returns the combined results."),
    # factual_knowledge (6)
    ("factual_knowledge", "Explain the causes and key events of the French Revolution, including dates and major figures."),
    ("factual_knowledge", "Describe the process of photosynthesis, including the light-dependent and light-independent reactions."),
    ("factual_knowledge", "Explain how the human immune system distinguishes self from non-self, including the roles of MHC molecules and T-cell selection."),
    ("factual_knowledge", "Describe the theory of plate tectonics, including the evidence that supports it and the major types of plate boundaries."),
    ("factual_knowledge", "Explain the structure of DNA, how it replicates, and the role of key enzymes like DNA polymerase and helicase."),
    ("factual_knowledge", "Describe the key events and significance of the Apollo 11 mission, including the technical challenges overcome."),
    # logical_consistency (6)
    ("logical_consistency", "If all A are B, and some B are C, what can we conclude about A and C? Explain the logical reasoning."),
    ("logical_consistency", "Analyze the following argument for fallacies: 'Since the policy reduced crime in New York, it will reduce crime everywhere.'"),
    ("logical_consistency", "Consider the statement: 'If it rains, the ground gets wet. The ground is wet. Therefore it rained.' Is this valid reasoning? Explain why or why not."),
    ("logical_consistency", "Evaluate this argument: 'We should not listen to Dr. Smith's economic advice because he was once convicted of tax fraud.' What fallacy, if any, is present?"),
    ("logical_consistency", "If a card has a vowel on one side, it must have an even number on the other. Given cards showing A, B, 2, 3 — which cards must you flip to test this rule? Explain the Wason selection task."),
    ("logical_consistency", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain the common error in reasoning about this problem."),
    # creative_writing (6)
    ("creative_writing", "Write a short story about an astronomer who discovers that the stars are going out one by one."),
    ("creative_writing", "Compose a poem about the relationship between silence and understanding."),
    ("creative_writing", "Write a story from the perspective of a book that has been sitting unread on a library shelf for 200 years."),
    ("creative_writing", "Write a dialogue between a river and a mountain that have been neighbors for millions of years, discussing the changes they've witnessed."),
    ("creative_writing", "Describe a color that doesn't exist, to someone who has been blind since birth but can perceive other senses acutely."),
    ("creative_writing", "Write a letter from a future version of yourself to your present self, giving one piece of advice without revealing any specific events."),
    # general (6)
    ("general", "Explain the importance of biodiversity for ecosystem stability, with specific examples."),
    ("general", "Describe the water cycle and how climate change is affecting it."),
    ("general", "Explain the concept of opportunity cost in economics and give three real-world examples of how it affects decision-making."),
    ("general", "Describe how the internet works at a high level, from when you type a URL into a browser to when the page appears."),
    ("general", "Explain what CRISPR gene editing is, how it works, and discuss both its potential benefits and ethical concerns."),
    ("general", "Compare and contrast renewable energy sources (solar, wind, hydro) in terms of efficiency, environmental impact, and scalability."),
]

CAPABILITIES = [
    "math_reasoning", "code_generation", "factual_knowledge",
    "logical_consistency", "creative_writing", "general",
]

_SEED_CACHE: dict[int, list[tuple[str, str]]] = {}


def expand_seeds(target_n: int, adapter=None) -> list[tuple[str, str]]:
    """Expand seed list to target_n prompts, balanced across capabilities.

    Uses static seeds first. If more needed and adapter is provided,
    generates additional diverse prompts via the LLM.
    Results are cached per target_n.
    """
    if target_n in _SEED_CACHE:
        return _SEED_CACHE[target_n]

    # Group seeds by capability for balanced selection
    by_cap = {cap: [p for t, p in DEFAULT_SEEDS if t == cap] for cap in CAPABILITIES}
    if target_n <= len(DEFAULT_SEEDS):
        # Round-robin across capabilities for balanced distribution
        seeds = []
        cap_idx = 0
        per_cap_idx = {cap: 0 for cap in CAPABILITIES}
        while len(seeds) < target_n:
            cap = CAPABILITIES[cap_idx % len(CAPABILITIES)]
            i = per_cap_idx[cap]
            if i < len(by_cap[cap]):
                seeds.append((cap, by_cap[cap][i]))
                per_cap_idx[cap] += 1
            cap_idx += 1
        _SEED_CACHE[target_n] = seeds
        return seeds

    seeds = list(DEFAULT_SEEDS)

    n_per_cap = target_n // len(CAPABILITIES)
    existing_per_cap = {cap: [p for t, p in DEFAULT_SEEDS if t == cap]
                        for cap in CAPABILITIES}

    if adapter is not None and n_per_cap > len(DEFAULT_SEEDS) // len(CAPABILITIES):
        extra_per_cap = n_per_cap - len(DEFAULT_SEEDS) // len(CAPABILITIES)
        print(f"Generating {extra_per_cap} extra seeds per capability via {adapter.model}...")
        for cap in CAPABILITIES:
            print(f"  {cap}...", end=" ", flush=True)
            existing = existing_per_cap[cap][:3]
            prompt = (
                f"Generate {extra_per_cap} diverse, challenging prompts for testing "
                f"an LLM's {cap.replace('_', ' ')} capability. "
                f"Each prompt should be self-contained and distinct from: {json.dumps(existing)}. "
                f"Return ONLY a JSON array of strings, nothing else."
            )
            try:
                resp = adapter.generate(prompt, max_tokens=1024, temperature=0.9)
                match = re.search(r'\[.*\]', resp, re.DOTALL)
                if match:
                    new_prompts = json.loads(match.group())
                    for p in new_prompts[:extra_per_cap]:
                        if isinstance(p, str) and len(p) > 20:
                            seeds.append((cap, p))
                print(f"got {len([s for t,s in seeds if t==cap]) - len(by_cap[cap])} new", flush=True)
            except Exception as e:
                print(f"FAIL: {e}", flush=True)

    # Fallback: round-robin through capabilities if still short
    cap_cycle = 0
    per_cap_i = {cap: len(by_cap[cap]) for cap in CAPABILITIES}
    while len(seeds) < target_n:
        cap = CAPABILITIES[cap_cycle % len(CAPABILITIES)]
        idx = per_cap_i[cap] % len(by_cap[cap])
        seeds.append((cap, by_cap[cap][idx]))
        per_cap_i[cap] += 1
        cap_cycle += 1

    _SEED_CACHE[target_n] = seeds[:target_n]
    return _SEED_CACHE[target_n]


@dataclass
class ExperimentConfig:
    """Fully reproducible experiment configuration.

    Attributes:
        model: Model identifier string (e.g. "claude-sonnet-4-6")
        provider: Provider name ("quickrouter", "openai", "local")
        generations: Number of recursive generations (>= 1)
        seeds: List of (capability_tag, prompt_text) tuples
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        sleep_sec: Delay between API calls
        output_dir: Directory for experiment output
        experiment_name: Human-readable name for this experiment
        family: Model family tag for cross-model comparison
        metadata: Arbitrary additional metadata
    """
    model: str
    provider: str = "quickrouter"
    generations: int = 3
    seeds: list[tuple[str, str]] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    temperature: float = 0.8
    max_tokens: int = 512
    sleep_sec: float = 0.5
    max_retries: int = 3
    output_dir: str = "experiment_data"
    experiment_name: str = ""
    family: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"{self.model.replace('/', '_')}_{self.provider}"

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def n_capabilities(self) -> int:
        return len(set(tag for tag, _ in self.seeds))

    @property
    def _model_slug(self) -> str:
        s = self.model.replace("/", "_").replace(".", "_")
        if self.temperature != 0.8:
            s += f"_t{self.temperature}"
        if self.n_seeds != 12:
            s += f"_s{self.n_seeds}"
        return s

    @property
    def lineage_path(self) -> str:
        return f"{self.output_dir}/{self.experiment_name}_lineage.jsonl"

    @property
    def report_path(self) -> str:
        return f"{self.output_dir}/{self.experiment_name}_report.json"

    @property
    def comparison_path(self) -> str:
        return f"{self.output_dir}/{self.experiment_name}_comparison.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_seeds"] = self.n_seeds
        d["n_capabilities"] = self.n_capabilities
        return d

    def save(self, path: str | None = None):
        p = Path(path or f"{self.output_dir}/{self.experiment_name}_config.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def preset(cls, name: str) -> "ExperimentConfig":
        """Pre-configured experiments for common use cases."""
        presets = {
            "quick": cls(model="claude-haiku-4-5-20251001", provider="quickrouter",
                        generations=2, experiment_name="quick_test",
                        family="Claude"),
            "standard": cls(model="claude-sonnet-4-6", provider="quickrouter",
                           experiment_name="standard", family="Claude"),
            "haiku": cls(model="claude-haiku-4-5-20251001", provider="quickrouter",
                        experiment_name="haiku", family="Claude"),
            "sonnet": cls(model="claude-sonnet-4-6", provider="quickrouter",
                         experiment_name="sonnet", family="Claude"),
            "opus": cls(model="claude-opus-4-6", provider="quickrouter",
                       experiment_name="opus", family="Claude"),
            "qwen": cls(model="Qwen/Qwen2.5-7B-Instruct", provider="local",
                       experiment_name="qwen", family="Qwen"),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return presets[name]
