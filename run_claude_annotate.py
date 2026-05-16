"""Claude proxy annotation via QuickRouter API. Uses urllib to avoid openai lib issues."""
import sys, os, json, time, re, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from synthetic_decay_monitor.constraint_extractor import HybridConstraintExtractor, extract_text_features

API_KEY = "sk-5iFgHQQRiOLyzqu7FUvVvrUahVadoqVUqNLjVEPUyzunh79e"
BASE_URL = "https://api.quickrouter.ai/v1"
MODEL = "claude-sonnet-4-6"

ANNOTATION_PROMPT = """You are evaluating text quality across three degradation dimensions.

Rate from 0 (severely degraded) to 5 (perfect) for each dimension:

E-I (Logic/Axiom): Clear reasoning chains, logical connectors like therefore/because/thus/hence. 5=flawless logic, 0=broken incoherent reasoning.
E-II (Style/Scale): Natural varied writing, rich vocabulary, no excessive repetition. 5=natural prose, 0=extreme repetition/filler-heavy.
E-III (Fact/Boundary): Accurate facts, proper capitalization, precise numbers. 5=accurate and properly formatted, 0=fabricated/garbled.

Reply with ONLY a JSON object on one line. No markdown, no explanation, no other text:
{"E-I": <int 0-5>, "E-II": <int 0-5>, "E-III": <int 0-5>}

TEXT:
{text}"""


def call_claude(prompt: str, max_tokens: int = 100) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def parse_json_safe(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pattern in [r'\{[^{}]*"E-I"[^{}]*\}', r'\{[^{}]*"E-II"[^{}]*\}', r'\{.*\}']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return {}


# Load texts from Qwen experiment
jsonl_path = "/Users/dengxinhang/paper/experiment_data/real_lineage.jsonl"
all_samples = []
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if line:
            all_samples.append(json.loads(line))

target_dims = {"factual_knowledge", "creative_writing"}
target_gens = {0, 2, 4}
selected = {}
for s in all_samples:
    dims = set(s.get("capability_tags", []))
    if dims & target_dims and s["generation"] in target_gens:
        dim = (dims & target_dims).pop()
        key = (dim, s["generation"])
        if key not in selected:
            selected[key] = []
        if len(selected[key]) < 2:
            selected[key].append(s)

texts_to_annotate = []
for (dim, gen), samples_list in sorted(selected.items()):
    for s in samples_list[:2]:
        texts_to_annotate.append({
            "id": s["id"], "text": s["text"], "generation": s["generation"],
            "dimension": dim,
        })

print(f"Annotating {len(texts_to_annotate)} texts with claude-sonnet-4-6...")
annotated = []
for item in texts_to_annotate:
    try:
        prompt = ANNOTATION_PROMPT.replace("{text}", item["text"][:1500])
        raw = call_claude(prompt, max_tokens=100)
        scores = parse_json_safe(raw)
        item["claude_ei"] = int(scores.get("E-I", -1))
        item["claude_eii"] = int(scores.get("E-II", -1))
        item["claude_eiii"] = int(scores.get("E-III", -1))
        print(f"  {item['id']}: EI={item['claude_ei']} EII={item['claude_eii']} EIII={item['claude_eiii']}  raw={repr(raw[:80])}")
    except Exception as e:
        print(f"  {item['id']}: ERROR {e}")
        item["claude_ei"] = item["claude_eii"] = item["claude_eiii"] = -1
    annotated.append(item)
    time.sleep(0.8)

# Compute DecayEngine features
extractor = HybridConstraintExtractor(judge_fn=None)
for item in annotated:
    state = extractor.extract_sample(item["text"])
    item["decay_Sn"] = (state.sigma_fact + state.sigma_syntax + state.sigma_style
                        + state.sigma_safety + state.sigma_coherence) / 5.0
    features = extract_text_features(item["text"])
    item["decay_ei"] = features.get("ei_logic_density", 0.5)
    item["decay_eii"] = features.get("eii_bigram_repetition", 0.5)
    item["decay_eiii"] = features.get("eiii_proper_case_ratio", 0.5)


def compute_spearman(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = {v: i + 1 for i, v in enumerate(sorted(set(x)))}
    rank_y = {v: i + 1 for i, v in enumerate(sorted(set(y)))}
    rx = [rank_x[v] for v in x]
    ry = [rank_y[v] for v in y]
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


print("\n=== Claude vs DecayEngine: Spearman rho ===")
for dim_name, feat_key, claude_key in [
    ("E-I (Logic)", "decay_ei", "claude_ei"),
    ("E-II (Style)", "decay_eii", "claude_eii"),
    ("E-III (Fact)", "decay_eiii", "claude_eiii"),
]:
    dec_vals = [a[feat_key] for a in annotated if a[claude_key] >= 0]
    claude_vals = [a[claude_key] for a in annotated if a[claude_key] >= 0]
    if len(dec_vals) >= 3:
        rho = compute_spearman(dec_vals, claude_vals)
        print(f"  {dim_name}: rho = {rho:+.3f} (n={len(dec_vals)})")
    else:
        print(f"  {dim_name}: insufficient valid samples (n={len(dec_vals)})")

for a in annotated:
    if a["claude_ei"] >= 0:
        a["claude_avg"] = (a["claude_ei"] + a["claude_eii"] + a["claude_eiii"]) / 3.0

dec_all = [a["decay_Sn"] for a in annotated if a.get("claude_ei", -1) >= 0]
claude_all = [(a["claude_ei"] + a["claude_eii"] + a["claude_eiii"]) / 3.0
              for a in annotated if a.get("claude_ei", -1) >= 0]
if len(dec_all) >= 3:
    rho_overall = compute_spearman(dec_all, claude_all)
    print(f"  Overall (Claude avg vs S_n): rho = {rho_overall:+.3f}")

out_path = "experiment_data/claude_annotation.json"
with open(out_path, "w") as f:
    json.dump([{k: v for k, v in a.items() if k != "text"} for a in annotated], f, indent=2)
print(f"\nSaved: {out_path}")
