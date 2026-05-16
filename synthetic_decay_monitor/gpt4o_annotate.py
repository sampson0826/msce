"""GPT-4o proxy annotation: rate Qwen texts on E-I/E-II/E-III, compare with DecayEngine S_n.

Computes Spearman rho between GPT-4o judgments and DecayEngine stability scores.
"""
import os, json, time
import numpy as np

from openai import OpenAI
from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor
from synthetic_decay_monitor.data_lineage import DatasetLineage, DataSample


ANNOTATION_PROMPT = """You are evaluating the quality of AI-generated text across three degradation dimensions.

Read the text below and rate it on each dimension from 0 (severely degraded) to 5 (perfect):

E-I (Logic/Axiom): Does the text have clear logical structure? Are reasoning chains coherent?
  - 5 = flawless logical flow, clear connectors (therefore, because, thus)
  - 0 = completely broken logic, incoherent reasoning

E-II (Style/Scale): Is the writing style natural and varied? No excessive repetition?
  - 5 = rich vocabulary, natural variation, no unnatural repetition
  - 0 = extreme repetition, truncated words, filler-heavy

E-III (Fact/Boundary): Are facts, proper names, and numbers accurate and properly formatted?
  - 5 = accurate facts, proper capitalization, precise numbers
  - 0 = fabricated facts, lowercase proper names, garbled numbers

Respond with exactly this JSON format, no other text:
{"E-I": <int 0-5>, "E-II": <int 0-5>, "E-III": <int 0-5>}

TEXT:
{text}
"""


def annotate_texts(client, texts_with_meta: list[dict]) -> list[dict]:
    """Get GPT-4o ratings for a batch of texts."""
    results = []
    for item in texts_with_meta:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ANNOTATION_PROMPT.format(text=item["text"][:1500])}],
                max_tokens=100, temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.strip("```json").strip("```").strip()
            scores = json.loads(raw)
            item["gpt4o_ei"] = int(scores.get("E-I", -1))
            item["gpt4o_eii"] = int(scores.get("E-II", -1))
            item["gpt4o_eiii"] = int(scores.get("E-III", -1))
            print(f"  {item['id']}: EI={item['gpt4o_ei']} EII={item['gpt4o_eii']} EIII={item['gpt4o_eiii']}")
        except Exception as e:
            print(f"  {item['id']}: ERROR {e}")
            item["gpt4o_ei"] = item["gpt4o_eii"] = item["gpt4o_eiii"] = -1
        results.append(item)
        time.sleep(0.3)
    return results


def compute_spearman(x, y):
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = {v: i+1 for i, v in enumerate(sorted(set(x)))}
    rank_y = {v: i+1 for i, v in enumerate(sorted(set(y)))}
    rx = [rank_x[v] for v in x]
    ry = [rank_y[v] for v in y]
    d2 = sum((a-b)**2 for a, b in zip(rx, ry))
    return 1 - 6*d2/(n*(n**2-1))


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Load 12 texts from Qwen experiment: 2 dimensions x 3 generations x 2 texts
    jsonl_path = "/Users/dengxinhang/paper/experiment_data/real_lineage.jsonl"
    all_samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    # Select: factual_knowledge + creative_writing, gens 0,2,4, 2 texts each = 12
    target_dims = {"factual_knowledge", "creative_writing"}
    target_gens = {0, 2, 4}
    selected = {}
    for s in all_samples:
        dims = set(s.get("capability_tags", []))
        if dims & target_dims and s["generation"] in target_gens:
            dim = (dims & target_dims).pop()
            key = (dim, s["generation"])
            if key not in selected or len(selected) < 12:
                if key not in selected:
                    selected[key] = []
                if len(selected[key]) < 2:
                    selected[key].append(s)

    texts_to_annotate = []
    for (dim, gen), samples in sorted(selected.items()):
        for s in samples[:2]:
            texts_to_annotate.append({
                "id": s["id"], "text": s["text"], "generation": s["generation"],
                "dimension": dim, "text": s["text"],
            })

    print(f"Annotating {len(texts_to_annotate)} texts with GPT-4o...")
    annotated = annotate_texts(client, texts_to_annotate)

    # Compute DecayEngine features for same texts
    extractor = HybridConstraintExtractor(judge_fn=None)
    for item in annotated:
        state = extractor.extract_sample(item["text"])
        item["decay_Sn"] = state.total_constraint
        item["decay_ei"] = state.text_features.get("ei_logic_density", 0.5)
        item["decay_eii"] = state.text_features.get("eii_bigram_repetition", 0.5)
        item["decay_eiii"] = state.text_features.get("eiii_proper_case_ratio", 0.5)

    # Compute Spearman rho
    print("\n=== GPT-4o vs DecayEngine: Spearman rho ===")
    for dim_name, feat_key, gpt4o_key in [
        ("E-I (Logic)", "decay_ei", "gpt4o_ei"),
        ("E-II (Style)", "decay_eii", "gpt4o_eii"),
        ("E-III (Fact)", "decay_eiii", "gpt4o_eiii"),
    ]:
        dec_vals = [a[feat_key] for a in annotated if a[gpt4o_key] >= 0]
        gpt_vals = [a[gpt4o_key] for a in annotated if a[gpt4o_key] >= 0]
        if len(dec_vals) >= 3:
            rho = compute_spearman(dec_vals, gpt_vals)
            print(f"  {dim_name}: rho = {rho:+.3f} (n={len(dec_vals)})")

    # Overall: GPT4o average score vs DecayEngine S_n
    for a in annotated:
        a["gpt4o_avg"] = (a["gpt4o_ei"] + a["gpt4o_eii"] + a["gpt4o_eiii"]) / 3.0
    dec_all = [a["decay_Sn"] for a in annotated if a["gpt4o_ei"] >= 0]
    gpt_all = [a["gpt4o_avg"] for a in annotated if a["gpt4o_ei"] >= 0]
    if len(dec_all) >= 3:
        rho_overall = compute_spearman(dec_all, gpt_all)
        print(f"  Overall (GPT4o avg vs S_n): rho = {rho_overall:+.3f}")

    # Save results
    out_path = "/Users/dengxinhang/paper/experiment_data/gpt4o_annotation.json"
    with open(out_path, "w") as f:
        json.dump([{k: v for k, v in a.items() if k != "text"} for a in annotated], f, indent=2)
    print(f"\nSaved: {out_path}")
