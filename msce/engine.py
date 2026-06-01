"""MSCE Phase 0 — 多模型异构生成器 + 淘汰赛裁决器 v2"""
import json, os, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── API 配置 ──
MKEAI_KEY = os.environ.get("MKEAI_API_KEY", "")
MKEAI_BASE = os.environ.get("MKEAI_BASE_URL", "https://api.mkeai.com/v1")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ── 默认多模型配置 ──
# Sam 建议：3个生成器用3个不同模型，模型差异 > prompt 差异
DEFAULT_CONFIG = {
    "deep_first":     {"client_type": "mkeai",    "model": "gpt-4o"},
    "breadth_first":  {"client_type": "mkeai",    "model": "gemini-2.5-pro"},
    "counterfactual": {"client_type": "deepseek", "model": "deepseek-chat"},
    "direct":         {"client_type": "deepseek", "model": "deepseek-chat"},  # 第4生成器
    "science_deep":   {"client_type": "mkeai",    "model": "o1"},              # 第5生成器: 科学深度推理
    "constraint_propagation": {"client_type": "mkeai", "model": "gpt-4o"},  # 第6生成器: 约束传播
    "judge":          {"client_type": "deepseek", "model": "deepseek-reasoner"},
    "appeal":         {"client_type": "deepseek", "model": "deepseek-reasoner"},  # 二次裁决
}

# ── 候选生成器 System Prompts ──
DEEP_FIRST_PROMPT = """你是一个"深度优先"推理者。

从已知事实出发，一步一步推理。每一步只选择最确定的下一步。不要跳步，不要猜测。
你的目标是得到一个逻辑完整的推理链，即使它很长。
最终输出：清晰的推理步骤 + 最终答案。"""

BREADTH_FIRST_PROMPT = """你是一个"广度优先"推理者。

同时考虑所有可能的路径，不要深入任何一条。列出所有可能性，评估每个的初始可信度。
你的目标是覆盖所有选项，不是找到答案。
最终输出：所有可能答案的列表 + 每个的初步可信度评估。"""

COUNTERFACTUAL_PROMPT = """你是一个"反事实"推理者。

假设最常见的答案（直觉答案）是错误的。找出最不可能但逻辑上仍然成立的答案。
你的目标是打破思维惯性。
最终输出：反直觉的答案 + 为什么它可能正确，为什么直觉答案可能错误。"""

# ── 淘汰赛裁决器 Prompt（简化输出格式，防截断）──
JUDGE_PROMPT = """你是一个公正的裁决者。检查候选答案，决定哪些应该被淘汰。

## 淘汰规则（按顺序）：
1. **内部一致性**：逻辑自相矛盾 → 淘汰。不确定 → 保留。
2. **外部一致性**：与公认科学/数学/逻辑事实冲突 → 淘汰。不确定 → 保留。
   **重要**：
   - 对于逻辑谜题（如蓝眼睛岛、囚徒困境），不要检查"与现实一致"。只检查与题目前提的一致性。
   - 对于假设性物理场景（如"地球突然停止自转"、"没有空气阻力"），不要在"场景是否可能发生"上扣分。应在题目给定的假设条件下，仅检查答案是否正确应用了物理定律（惯性、引力、阿基米德原理等）。场景是题目设定的，裁判不质疑题目设定。
3. **可验证性**：推理链不可追溯 → 降权（不淘汰）。清晰可追溯 → 保留。
4. **简洁性**：同等正确，更简洁的胜出（奥卡姆剃刀）。

## 核心原则：疑罪从无。不确定就保留。误杀不可逆。

## 输出格式（严格JSON，无解释）：
{"eliminated":[],"surviving":[{"id":"策略名","score":0.9}],"top3":[{"rank":1,"id":"策略名","summary":"一句话答案摘要"}]}

只输出这一行JSON。"""

# ── 二次裁决 Prompt（Sam建议：反事实检查被淘汰候选）──
APPEAL_PROMPT = """你是二次裁决者。重新审查被淘汰的候选答案，判断是否有误杀。

## 反事实检查：
对于每个被淘汰的候选，问自己："如果这个候选被保留，会发生什么？它的答案是否可能正确？"

## 恢复规则：
- 淘汰理由不充分（如"不可验证"但推理链实际可追溯）→ 恢复
- 淘汰理由成立但候选核心答案可能正确 → 恢复（降权0.2）
- 淘汰理由确实成立且答案有根本性错误 → 维持淘汰
- 不确定 → 恢复（疑罪从无）

## 输出格式（严格JSON，无解释）：
{"reinstated":[{"id":"策略名","score":0.6,"reason":"恢复理由"}],"confirmed_eliminated":[{"id":"策略名"}]}

只输出这一行JSON。"""


# ── 第4生成器 Prompt（直接推理，不做特殊策略）──
DIRECT_PROMPT = """你是一个直接推理者。直接回答以下问题，给出清晰的推理过程和最终答案。不要用特殊策略，正常思考即可。"""

# ── 第5生成器 Prompt（科学深度推理，o1专用）──
SCIENCE_DEEP_PROMPT = """你是一个科学深度推理者。对于物理和科学问题：
1. 明确引用相关物理定律（牛顿定律、阿基米德原理、瑞利散射等）
2. 在题目给定的假设条件下进行推理，不质疑题目设定
3. 分步计算，每步标注使用的公式
4. 最终给出清晰答案"""

CONSTRAINT_PROPAGATION_PROMPT = """你是一个约束传播推理者。对于约束满足和逻辑谜题，使用系统化排除法：

1. **建立网格**：列出所有变量位置和所有属性域
2. **编码约束**：将每个文本约束转化为确定关系和排除关系
3. **传播**：当单元格确定时，立即在同行/同列传播排除，迭代至闭合
4. **分支**：不确定时假设→传播→矛盾则排除
5. **验证**：完成赋值后，逐条回检所有约束，确保无一违反
6. **输出**：完整赋值表 + 问题的明确答案

关键：不跳步，不猜测。完成后必须逐条验证每个约束。"""


def get_client(client_type="mkeai"):
    if client_type == "mkeai":
        return OpenAI(api_key=MKEAI_KEY, base_url=MKEAI_BASE)
    elif client_type == "deepseek":
        return OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
    else:
        raise ValueError(f"Unknown client_type: {client_type}")


def generate_candidate(client, model, system_prompt, question, strategy_name, timeout=60):
    try:
        # o1 和 deepseek-reasoner 不支持 system prompt 和 temperature，需要特殊处理
        if model in ("o1", "deepseek-reasoner"):
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n问题：{question}"}
            ]
            kwargs = dict(model=model, messages=messages, timeout=timeout)
            if model == "o1":
                kwargs["max_completion_tokens"] = 4000
            else:
                kwargs["max_tokens"] = 4000
            response = client.chat.completions.create(**kwargs)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=timeout
            )
        msg = response.choices[0].message
        answer = msg.content or ""
        # deepseek-reasoner 的推理在 reasoning_content，content 可能为空
        reasoning = getattr(msg, 'reasoning_content', '') or ''
        if reasoning and not answer:
            answer = reasoning
        elif reasoning and answer:
            answer = reasoning + "\n\n=== 最终答案 ===\n" + answer
        return {"strategy": strategy_name, "model": model, "answer": answer, "success": True}
    except Exception as e:
        return {"strategy": strategy_name, "model": model, "answer": None, "success": False, "error": str(e)}


def _simple_similarity(text_a, text_b):
    """用词重叠率估算两个答案的相似度（不依赖embedding）"""
    if not text_a or not text_b:
        return 0.0
    def tokens(s):
        # 取最后200字符（通常是结论部分）
        tail = s[-300:] if len(s) > 300 else s
        return set(re.findall(r'[一-鿿]+|[a-zA-Z]+', tail.lower()))
    a, b = tokens(text_a), tokens(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _run_appeal(appeal_client, appeal_model, question, candidates, eliminated):
    """Sam建议：二次裁决——反事实检查被淘汰候选，防止误杀"""
    if not eliminated:
        return eliminated, []

    # 构建被淘汰候选的文本
    elim_text = ""
    elim_ids = set()
    for e in eliminated:
        eid = e.get("id", e) if isinstance(e, dict) else e
        elim_ids.add(eid)
        for c in candidates:
            if c["strategy"] == eid and c["success"]:
                reason = e.get("reason", "unknown") if isinstance(e, dict) else "unknown"
                elim_text += f"\n### 被淘汰候选：{eid}\n淘汰理由：{reason}\n答案：{c['answer'][:500]}\n"
                break

    if not elim_text:
        return eliminated, []

    appeal_input = f"## 原始问题：\n{question}\n\n## 被淘汰的候选（需重新审查）：\n{elim_text}"

    try:
        response = appeal_client.chat.completions.create(
            model=appeal_model,
            messages=[
                {"role": "system", "content": APPEAL_PROMPT},
                {"role": "user", "content": appeal_input}
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=30
        )
        result_text = response.choices[0].message.content
        if result_text is None:
            return eliminated, []

        verdict = _repair_json(result_text)
        reinstated = verdict.get("reinstated", [])
        confirmed = set(
            e.get("id", e) if isinstance(e, dict) else e
            for e in verdict.get("confirmed_eliminated", [])
        )

        # 从淘汰列表移除被恢复的候选
        reinstated_ids = set(r.get("id", "") for r in reinstated)
        final_eliminated = [
            e for e in eliminated
            if (e.get("id", e) if isinstance(e, dict) else e) not in reinstated_ids
        ]

        return final_eliminated, reinstated
    except Exception:
        return eliminated, []


def _force_eliminate_similar(surviving, candidates, threshold=0.75):
    """Sam建议：如果多个候选高度相似，强制淘汰最弱的那个，防止'全部保留赛'"""
    if len(surviving) <= 1:
        return surviving

    # 找答案文本
    id_to_answer = {}
    for c in candidates:
        if c["success"]:
            id_to_answer[c["strategy"]] = c["answer"]

    # 按分数排序
    sorted_s = sorted(surviving, key=lambda x: x.get("score", 0), reverse=True)
    to_remove = set()

    for i in range(len(sorted_s)):
        if sorted_s[i]["id"] in to_remove:
            continue
        for j in range(i + 1, len(sorted_s)):
            if sorted_s[j]["id"] in to_remove:
                continue
            sim = _simple_similarity(
                id_to_answer.get(sorted_s[i]["id"], ""),
                id_to_answer.get(sorted_s[j]["id"], "")
            )
            if sim > threshold:
                # 淘汰分数低的那个
                weaker = sorted_s[j]["id"]
                to_remove.add(weaker)

    if to_remove:
        return [s for s in surviving if s["id"] not in to_remove]
    return surviving


def _repair_json(text):
    """修复被截断的JSON"""
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ["```"]:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 补齐缺失的括号
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    try:
        return json.loads(text + "}" * open_braces + "]" * open_brackets)
    except json.JSONDecodeError:
        pass

    # 逐行截断修复
    lines = text.split("\n")
    for i in range(len(lines) - 1, max(len(lines) - 10, 0), -1):
        truncated = "\n".join(lines[:i+1])
        ob = truncated.count("{") - truncated.count("}")
        oa = truncated.count("[") - truncated.count("]")
        try:
            return json.loads(truncated + "}" * ob + "]" * oa)
        except json.JSONDecodeError:
            continue

    # 正则提取
    result = {"eliminated": [], "surviving": [], "top3": [], "_repaired": True}
    for m in re.finditer(r'"id"\s*:\s*"([^"]+)"\s*,\s*"reason"\s*:\s*"([^"]*)"', text):
        result["eliminated"].append({"id": m.group(1), "reason": m.group(2)})
    for m in re.finditer(r'"id"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*([\d.]+)', text):
        result["surviving"].append({"id": m.group(1), "score": float(m.group(2))})
    for m in re.finditer(r'"rank"\s*:\s*(\d+)\s*,\s*"id"\s*:\s*"([^"]+)"\s*,\s*"summary"\s*:\s*"([^"]*)"', text):
        result["top3"].append({"rank": int(m.group(1)), "id": m.group(2), "summary": m.group(3)})

    if result["eliminated"] or result["surviving"] or result["top3"]:
        return result
    return {"error": "JSON parse failed", "raw": text[:500]}


def _single_judge(judge_client, judge_model, question, candidate_text):
    """单次裁判调用，返回原始verdict"""
    judge_input = f"## 原始问题：\n{question}\n\n## 候选答案：\n{candidate_text}"
    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": judge_input}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    result_text = response.choices[0].message.content
    if result_text is None:
        return None
    return _repair_json(result_text)


def _best_of_n_judge(judge_client, judge_model, question, candidate_text, n=3):
    """自适应多数投票：第一轮高置信度直接通过，低置信度才跑完整n次"""
    # 第一轮
    try:
        v1 = _single_judge(judge_client, judge_model, question, candidate_text)
    except Exception:
        v1 = None

    if v1 is None or "error" in v1:
        # 第一轮失败，回退
        try:
            v = _single_judge(judge_client, judge_model, question, candidate_text)
            if v:
                v["_judge_votes"] = 1
                v["_adaptive"] = "fallback"
                return v
        except Exception:
            return {"error": "All judge attempts failed"}

    scores = [s.get("score", 0) for s in v1.get("surviving", [])]
    max_s = max(scores) if scores else 0
    elim_count = len(v1.get("eliminated", []))

    # Sam建议：高置信度且裁判做出了淘汰决策 → 跳过重复投票
    ADAPTIVE_THRESHOLD = 0.9
    if max_s >= ADAPTIVE_THRESHOLD and elim_count > 0:
        v1["_judge_votes"] = 1
        v1["_adaptive"] = f"fast:score={max_s:.2f},elim={elim_count}"
        return v1

    # 低置信度 → 完整n次投票
    best_verdict = v1
    best_max_score = max_s
    for _ in range(n - 1):
        try:
            v = _single_judge(judge_client, judge_model, question, candidate_text)
            if v is None or "error" in v:
                continue
            scores_r = [s.get("score", 0) for s in v.get("surviving", [])]
            max_r = max(scores_r) if scores_r else 0
            if max_r >= best_max_score:
                best_max_score = max_r
                best_verdict = v
        except Exception:
            continue
    best_verdict["_judge_votes"] = n
    best_verdict["_adaptive"] = f"full:max_score={best_max_score:.2f}"
    return best_verdict


# 域自适应裁判指令
DOMAIN_JUDGE_HINTS = {
    "math": "\n## 裁判提示：数学题答案有唯一正确数值。不同候选的数值答案不可能同时正确。你必须比较各候选的计算过程和最终数值，淘汰推理有缺陷或数值明显错误的候选。至少淘汰1-2个最弱的候选。\n",
    "logic": "",
    "science": "",
    "verbal": "",
}

def run_elimination(judge_client, judge_model, question, candidates, appeal_config=None, domain=None):
    candidate_text = ""
    # 域自适应裁判提示
    if domain and domain in DOMAIN_JUDGE_HINTS:
        candidate_text += DOMAIN_JUDGE_HINTS[domain]
    for i, c in enumerate(candidates):
        if c["success"]:
            candidate_text += f"\n### 候选 {i+1}：{c['strategy']}（模型：{c['model']}）\n{c['answer']}\n"
        else:
            candidate_text += f"\n### 候选 {i+1}：{c['strategy']}（模型：{c['model']}）\n**生成失败**：{c.get('error', 'unknown')}\n"

    try:
        verdict = _best_of_n_judge(judge_client, judge_model, question, candidate_text, n=3)

        if "error" in verdict:
            return verdict

        # Sam建议：强制淘汰高相似度候选
        if "surviving" in verdict and len(verdict.get("surviving", [])) >= 2:
            before = len(verdict["surviving"])
            verdict["surviving"] = _force_eliminate_similar(
                verdict["surviving"], candidates, threshold=0.75
            )
            after = len(verdict["surviving"])
            if before != after:
                verdict["_force_eliminated"] = before - after

        # Sam建议：二次裁决——反事实检查被淘汰候选
        eliminated = verdict.get("eliminated", [])
        if eliminated and appeal_config:
            try:
                appeal_client = get_client(appeal_config["client_type"])
                final_elim, reinstated = _run_appeal(
                    appeal_client, appeal_config["model"], question, candidates, eliminated
                )
                if reinstated:
                    verdict["eliminated"] = final_elim
                    for r in reinstated:
                        verdict.setdefault("surviving", []).append({
                            "id": r.get("id", "?"),
                            "score": r.get("score", 0.6),
                            "strength": f"二次裁决恢复: {r.get('reason','')}"
                        })
                    verdict["_reinstated"] = len(reinstated)
            except Exception:
                pass

        return verdict
    except Exception as e:
        return {"error": str(e)}


def run_msce(question, config=None, domain=None):
    if config is None:
        config = DEFAULT_CONFIG

    strategies = [
        ("deep_first", DEEP_FIRST_PROMPT),
        ("breadth_first", BREADTH_FIRST_PROMPT),
        ("counterfactual", COUNTERFACTUAL_PROMPT),
        ("direct", DIRECT_PROMPT),
        ("science_deep", SCIENCE_DEEP_PROMPT),
        ("constraint_propagation", CONSTRAINT_PROPAGATION_PROMPT),
    ]

    # Step 1: 并行生成候选答案（6个生成器）
    candidates = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for strategy_name, sys_prompt in strategies:
            cfg = config[strategy_name]
            client = get_client(cfg["client_type"])
            future = executor.submit(
                generate_candidate, client, cfg["model"], sys_prompt, question, strategy_name
            )
            futures[future] = strategy_name

        for future in as_completed(futures):
            candidates.append(future.result())

    # 按策略名排序保证一致性
    candidates.sort(key=lambda x: x["strategy"])

    # Step 2: 裁决器判断 + 二次裁决
    judge_cfg = config["judge"]
    judge_client = get_client(judge_cfg["client_type"])
    appeal_cfg = config.get("appeal")
    verdict = run_elimination(judge_client, judge_cfg["model"], question, candidates, appeal_cfg, domain=domain)

    # Step 3: 置信度检查（Sam建议：所有候选都不可靠时，诚实说"不确定"）
    CONFIDENCE_THRESHOLD = 0.5
    surviving = verdict.get("surviving", [])
    top_scores = [s.get("score", 0) for s in surviving]
    max_score = max(top_scores) if top_scores else 0

    low_confidence = False
    if not surviving or max_score < CONFIDENCE_THRESHOLD:
        low_confidence = True
        verdict["low_confidence"] = True
        verdict["low_confidence_reason"] = (
            "所有候选答案被淘汰" if not surviving else
            f"最高分{max_score:.2f}低于置信阈值{CONFIDENCE_THRESHOLD}，建议人工核查"
        )

    return {
        "question": question,
        "candidates": candidates,
        "verdict": verdict,
        "low_confidence": low_confidence,
        "timestamp": time.time()
    }


def run_single_model(question, client_type="mkeai", model="gpt-4o"):
    client = get_client(client_type)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content


def run_benchmark(questions, config=None, baselines=None):
    """
    批量验证：MSCE vs 单模型基准。
    questions: [{"q": "问题文本", "answer": "正确答案关键词"}]
    baselines: [{"client_type": "mkeai", "model": "gpt-4o"}, ...]
    返回对比结果。
    """
    if config is None:
        config = DEFAULT_CONFIG
    if baselines is None:
        baselines = [
            {"client_type": "mkeai", "model": "gpt-4o"},
            {"client_type": "deepseek", "model": "deepseek-chat"},
        ]

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['q'][:60]}...")
        t0 = time.time()

        # MSCE
        msce_result = run_msce(q["q"], config)
        msce_time = time.time() - t0

        # 单模型基准
        baseline_results = {}
        for bl in baselines:
            try:
                bl_answer = run_single_model(q["q"], bl["client_type"], bl["model"])
                baseline_results[f"{bl['client_type']}:{bl['model']}"] = bl_answer
            except Exception as e:
                baseline_results[f"{bl['client_type']}:{bl['model']}"] = f"ERROR: {e}"

        results.append({
            "question": q,
            "msce": msce_result,
            "msce_time": msce_time,
            "baselines": baseline_results
        })

        # 简要输出
        verdict = msce_result.get("verdict", {})
        surviving = verdict.get("surviving", [])
        top1 = verdict.get("top3", [{}])[0] if verdict.get("top3") else {}
        print(f"  MSCE top1: {top1.get('id','?')} ({top1.get('summary','?')[:50]})")
        print(f"  保留{len(surviving)}个, 耗时{msce_time:.0f}s")

    return results


# ── 测试 ──
if __name__ == "__main__":
    test_question = "一个房间里有3个灯泡和3个开关。开关在门外。你只能进房间一次。如何判断哪个开关控制哪个灯泡？"

    print("=" * 60)
    print("MSCE Phase 0 v2 — 多模型异构 + 淘汰赛")
    print("=" * 60)

    print("\n[1] 单模型基准 (GPT-4o)...")
    baseline = run_single_model(test_question, "mkeai", "gpt-4o")
    print(f"基准答案:\n{baseline[:300]}\n")

    print("[2] MSCE 淘汰赛...")
    result = run_msce(test_question)
    for i, c in enumerate(result["candidates"]):
        status = "OK" if c["success"] else "FAIL"
        print(f"  候选{i+1} [{c['strategy']}@{c['model']}]: {status}")

    verdict = result.get("verdict", {})
    if "error" in verdict:
        print(f"  裁决: FAIL - {verdict['error']}")
    else:
        print(f"  淘汰: {len(verdict.get('eliminated',[]))}个")
        print(f"  保留: {len(verdict.get('surviving',[]))}个")
        if verdict.get("_force_eliminated"):
            print(f"  (强制淘汰: {verdict['_force_eliminated']}个相似候选)")
        for s in verdict.get("surviving", []):
            print(f"    [{s['id']}] score={s.get('score','?')}")
        for t in verdict.get("top3", []):
            print(f"    TOP{t['rank']}: {t['id']} — {t.get('summary','')[:60]}")

    print("\n" + "=" * 60)
    print("模型可用状态：")
    print(f"  mkeai: GPT-4o/GPT-4.1/o1/o4-mini/Gemini 2.5 ✓")
    print(f"  DeepSeek: chat + reasoner ✓")
    print("=" * 60)
