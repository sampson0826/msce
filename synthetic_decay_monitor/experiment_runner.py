"""Unified experiment runner for recursive generation + decay analysis.

Replaces 3 temporary generation scripts (run_claude_experiment.py,
run_haiku_experiment.py, run_opus_experiment.py) with a single interface.

Usage:
    python -m synthetic_decay_monitor.experiment_runner \
        --model claude-sonnet-4-6 --provider quickrouter

    python -m synthetic_decay_monitor.experiment_runner \
        --preset opus
"""
import sys, os, json, time
from pathlib import Path
from typing import Optional

from .provider_adapter import create_provider, ProviderAdapter
from .experiment_config import ExperimentConfig, DEFAULT_SEEDS, expand_seeds
from .data_lineage import DataSample, DatasetLineage, parse_lineage_from_jsonl
from .constraint_extractor import HybridConstraintExtractor
from .decay_engine import DecayEngine
from .executor_classifier import diagnose_executor_decay
from .report import generate_json_report


class ExperimentRunner:
    """Orchestrates a full recursive generation experiment.

    Pipeline:
    1. Generate: recursive model calls (Gen0 seeds → Gen1-3 responses)
    2. Save: output as JSONL lineage file
    3. Analyze: DecayEngine → trajectories, diagnoses, β
    4. Report: print summary + save JSON report
    5. Compare: cross-model comparison if baseline lineage files available
    """

    def __init__(self, config: ExperimentConfig, adapter: ProviderAdapter | None = None):
        self.config = config
        self.adapter = adapter or create_provider(
            config.provider, model=config.model, temperature=config.temperature
        )
        self._samples: list[DataSample] = []
        self._lineage: DatasetLineage | None = None
        self._trajectories: list = []
        self._collapse_order: list = []
        self._diagnoses: list = []
        self._global_beta: float = 0.0

    # ── Generation ─────────────────────────────────────────────

    def run_generation(self) -> DatasetLineage:
        """Execute recursive generation and return lineage."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._samples = []

        # Gen 0: seed prompts
        prev_texts = [p[1] for p in self.config.seeds]
        prev_tags = [[p[0]] for p in self.config.seeds]
        for i, (tags, text) in enumerate(zip(prev_tags, prev_texts)):
            self._samples.append(DataSample(
                text=text, generation=0, source_model="human",
                capability_tags=tags, sample_id=f"G0_{i:04d}",
            ))

        # Gen 1..N: recursive calls
        for gen in range(1, self.config.generations + 1):
            print(f"\n[Gen {gen}] {self.adapter.model} generating {len(prev_texts)} responses...")
            curr_texts = []
            for i, text in enumerate(prev_texts):
                resp = None
                for attempt in range(self.config.max_retries):
                    try:
                        resp = self.adapter.generate(text, max_tokens=self.config.max_tokens)
                        break
                    except Exception as e:
                        if attempt < self.config.max_retries - 1:
                            wait = 2 ** (attempt + 1)
                            print(f"  [{i+1}/{len(prev_texts)}] retry {attempt+1}/{self.config.max_retries} in {wait}s: {e}")
                            time.sleep(wait)
                        else:
                            print(f"  [{i+1}/{len(prev_texts)}] FAIL after {self.config.max_retries} retries: {e}")
                if resp is not None:
                    curr_texts.append(resp)
                    print(f"  [{i+1}/{len(prev_texts)}] {len(resp)} chars")
                else:
                    curr_texts.append(text)
                time.sleep(self.config.sleep_sec)

            for i, text in enumerate(curr_texts):
                self._samples.append(DataSample(
                    text=text, generation=gen,
                    source_model=self.config.model,
                    capability_tags=prev_tags[i],
                    sample_id=f"G{gen}_{i:04d}",
                ))
            prev_texts = curr_texts

        self._lineage = DatasetLineage(samples=self._samples)
        return self._lineage

    # ── Save ───────────────────────────────────────────────────

    def save_lineage(self) -> str:
        """Save lineage as JSONL. Returns path."""
        if self._lineage is None:
            raise RuntimeError("Run run_generation() first")
        path = self.config.lineage_path
        with open(path, "w") as f:
            for s in sorted(self._lineage.samples,
                           key=lambda x: (x.generation, x.sample_id)):
                f.write(json.dumps({
                    "id": s.sample_id, "text": s.text,
                    "generation": s.generation,
                    "source_model": s.source_model,
                    "capability_tags": s.capability_tags,
                }, ensure_ascii=False) + "\n")
        print(f"Saved lineage: {path} ({len(self._lineage.samples)} samples)")
        return path

    # ── Analysis ───────────────────────────────────────────────

    def run_analysis(self):
        """Run DecayEngine analysis on generated lineage."""
        if self._lineage is None:
            raise RuntimeError("Run run_generation() first")

        print("\n[Analysis] Running DecayEngine...")
        extractor = HybridConstraintExtractor(judge_fn=None)
        engine = DecayEngine(self._lineage, extractor)
        engine.run_all_capabilities()

        self._trajectories = engine.get_all_trajectories()
        self._collapse_order = engine.get_collapse_order()
        self._diagnoses = []
        for cap, snapshots in engine._snapshots.items():
            diag = diagnose_executor_decay(
                engine._trajectories.get(cap, []), snapshots, capability=cap
            )
            self._diagnoses.append(diag)

        betas = []
        for t in self._trajectories:
            if "trajectory" in t:
                for g in t["trajectory"]:
                    if "beta" in g and g["beta"] > 0:
                        betas.append(g["beta"])
        self._global_beta = sum(betas) / len(betas) if betas else 0.0

    def _engine(self):
        """Re-create engine for post-hoc analysis (used by comparison methods)."""
        if self._lineage is None:
            raise RuntimeError("Run run_generation() first")
        extractor = HybridConstraintExtractor(judge_fn=None)
        return DecayEngine(self._lineage, extractor)

    # ── Reporting ──────────────────────────────────────────────

    def print_trajectories(self):
        print(f"\n=== {self.config.model} Stability Trajectories ===")
        for t in self._trajectories:
            if "trajectory" in t and t["trajectory"]:
                last = t["trajectory"][-1]
                cap = t.get("capability", "?")
                print(f"  {cap}: S_{last['generation']}={last['S_n']:.3f} "
                      f"beta={last['beta']:.3f} [{last.get('status','?')}]"
                      f" -> collapse gen {t.get('predicted_collapse_gen', -1)}")
        if self._collapse_order:
            names = [c["capability"] if isinstance(c, dict) else str(c) for c in self._collapse_order]
            print(f"\n  Collapse order: {' > '.join(names)}")

    def print_diagnoses(self):
        print()
        for d in self._diagnoses:
            cap = d.get("capability", d.get("_cap", "?"))
            diag = d["diagnosis"]
            print(f"  {cap}: {diag.degradation_type} (sev={diag.severity:.2f})")

    def print_summary(self):
        print(f"\nGlobal beta ({self.config.model}): {self._global_beta:.4f}")
        print(f"  Provider: {self.adapter.provider_name}")
        print(f"  Seeds: {self.config.n_seeds} × {self.config.generations} gens "
              f"= {self.config.n_seeds * (self.config.generations + 1)} samples")

    def save_report(self) -> str:
        if self._lineage is None:
            raise RuntimeError("Run run_generation() first")
        json_report = generate_json_report(
            lineage=self._lineage,
            trajectories=self._trajectories,
            diagnoses=self._diagnoses,
        )
        path = self.config.report_path
        with open(path, "w") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        print(f"Saved report: {path}")
        return path

    def save_config(self):
        self.config.save()
        print(f"Saved config: {self.config.output_dir}/{self.config.experiment_name}_config.json")

    # ── Cross-model comparison ─────────────────────────────────

    def compare_with(self, other_lineage_paths: list[str],
                     labels: list[str] | None = None) -> dict:
        """Compare β across multiple models given their lineage JSONL files.

        Returns comparison dict with model betas, family stats.
        """
        if labels is None:
            labels = [Path(p).stem.replace("_lineage", "") for p in other_lineage_paths]

        models = []
        # Add self
        models.append({
            "name": self.config.model,
            "beta": self._global_beta,
            "family": self.config.family or "unknown",
            "label": self.config.experiment_name,
        })

        # Add others
        for path, label in zip(other_lineage_paths, labels):
            lin = parse_lineage_from_jsonl(path)
            eng = DecayEngine(lin, HybridConstraintExtractor(judge_fn=None))
            eng.run_all_capabilities()
            betas = []
            for t in eng.get_all_trajectories():
                if "trajectory" in t:
                    for g in t["trajectory"]:
                        if "beta" in g and g["beta"] > 0:
                            betas.append(g["beta"])
            other_beta = sum(betas) / len(betas) if betas else 0.0
            models.append({
                "name": label,
                "beta": other_beta,
                "family": "unknown",
                "label": label,
            })

        return self._format_comparison(models)

    def _format_comparison(self, models: list[dict]) -> dict:
        n_models = len(models)
        all_betas = [m["beta"] for m in models]
        mean_beta = sum(all_betas) / n_models
        import numpy as np
        std_beta = float(np.std(all_betas))

        baseline_beta = all_betas[0]  # first model as baseline

        print(f"\n{'='*60}")
        print(f"  {'Model':<22} {'β':>8} {'Family':>10} {'vs Baseline':>12}")
        print(f"  {'-'*48}")
        for m in models:
            print(f"  {m['name']:<22} {m['beta']:8.4f} {m['family']:>10} "
                  f"{m['beta']/baseline_beta:11.2f}x")

        # Family analysis
        families = {}
        for m in models:
            fam = m.get("family", "unknown")
            families.setdefault(fam, []).append(m["beta"])

        family_results = {}
        for fam, betas in families.items():
            if len(betas) > 1:
                rng = max(betas) - min(betas)
                within_02 = rng <= 0.02
                print(f"\n  {fam} family: mean β={sum(betas)/len(betas):.4f} "
                      f"range={rng:.4f} (n={len(betas)}) "
                      f"[{'PASS' if within_02 else 'FAIL'} β-constant]")
                family_results[fam] = {
                    "mean": sum(betas) / len(betas),
                    "range": rng,
                    "within_0_02": within_02,
                    "n": len(betas),
                }

        print(f"\n  All models: mean β={mean_beta:.4f} ± {std_beta:.4f}")

        return {
            "models": models,
            "mean_beta": mean_beta,
            "std_beta": std_beta,
            "n_models": n_models,
            "family_tests": family_results,
        }

    # ── Full pipeline ──────────────────────────────────────────

    def run(self, compare_paths: list[str] | None = None,
            compare_labels: list[str] | None = None) -> dict:
        """Run full pipeline: generate → analyze → report → compare.

        Returns dict with experiment summary.
        """
        self.run_generation()
        self.save_lineage()
        self.run_analysis()
        self.print_trajectories()
        self.print_diagnoses()
        self.print_summary()
        self.save_report()
        self.save_config()

        result = {
            "model": self.config.model,
            "provider": self.config.provider,
            "global_beta": self._global_beta,
            "n_samples": len(self._samples),
            "lineage_path": self.config.lineage_path,
            "report_path": self.config.report_path,
        }

        if compare_paths:
            result["comparison"] = self.compare_with(compare_paths, compare_labels)

        print("\nDone.")
        return result


# ── CLI ────────────────────────────────────────────────────────

def main():
    """CLI entry point for `decay-eval` command."""
    import argparse

    p = argparse.ArgumentParser(
        description="Decay Eval — measure LLM recursive stability via constraint residual beta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  decay-eval --preset qwen
  decay-eval --model deepseek-chat --provider deepseek
  decay-eval --preset sonnet --generations 5 --seeds 12
        """,
    )
    p.add_argument("--preset", choices=["quick", "standard", "haiku", "sonnet", "opus", "qwen"],
                   help="Use a preset configuration")
    p.add_argument("--model", help="Model ID (overrides preset)")
    p.add_argument("--provider", help="Provider name (overrides preset)")
    p.add_argument("--generations", type=int, default=3, help="Number of recursive generations")
    p.add_argument("--seeds", type=int, default=36, help="Number of seed prompts")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--output-dir", default="experiment_data", help="Output directory")
    p.add_argument("--compare", nargs="*", help="Lineage JSONL files to compare against")
    p.add_argument("--sleep-sec", type=float, default=0.5, help="Delay between API calls")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per API call")
    p.add_argument("--no-analyze", action="store_true", help="Skip analysis, only generate")

    args = p.parse_args()

    if args.preset:
        config = ExperimentConfig.preset(args.preset)
        if args.model:
            config.model = args.model
        if args.provider:
            config.provider = args.provider
    else:
        config = ExperimentConfig(
            model=args.model or "claude-sonnet-4-6",
            provider=args.provider or "quickrouter",
            generations=args.generations,
            output_dir=args.output_dir,
        )

    if args.generations != 3:
        config.generations = args.generations
    if args.temperature != 0.8:
        config.temperature = args.temperature
    config.sleep_sec = args.sleep_sec
    config.max_retries = args.max_retries

    # Wire --seeds: create adapter early for seed expansion if needed
    adapter = create_provider(config.provider, model=config.model)
    if args.seeds != len(config.seeds):
        config.seeds = expand_seeds(args.seeds, adapter=adapter)
        print(f"Expanded seeds: {len(config.seeds)} prompts "
              f"({len(set(t for t, _ in config.seeds))} capabilities)")

    runner = ExperimentRunner(config, adapter=adapter)
    result = runner.run(compare_paths=args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
