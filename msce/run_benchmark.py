"""MSCE Phase 0 — 20题验证 v2: 数值提取 + 可靠评分"""
import sys, json, time, re, os
sys.path.insert(0, '/Users/dengxinhang/paper/constraint_residual/msce')
from engine import run_msce, run_single_model, get_client
from benchmark_questions import BENCHMARK

def extract_number(text):
    """智能提取最终答案数字：优先匹配'最终答案'标记、单位后缀、等式结果"""
    # 1. 优先：**数字** 加粗格式（通常是最终答案）
    m = re.search(r'\*\*(\d+(?:\.\d+)?)\s*(?:立方|平方|厘米|小时|人|个|只|次|天)?\*\*', text)
    if m: return m.group(1)
    # 2. "最终答案" 后的数字
    m = re.search(r'最终答案[：:].*?(\d+(?:\.\d+)?)', text)
    if m: return m.group(1)
    # 3. "= 数字" 模式（等式结果）
    matches = re.findall(r'=\s*(\d+(?:\.\d+)?)\s*(?:立方|平方|厘米|小时|人|个|只|次|天)?', text)
    if matches: return matches[-1]
    # 4. 末尾数字（带单位优先）
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:立方|平方|厘米|小时|人|个|只|次|天)\s*[。\n]', text[-500:])
    if m: return m.group(1)
    # 5. 最后出现的合理数字
    numbers = re.findall(r'\d+(?:\.\d+)?', text[-500:])
    for n in reversed(numbers):
        v = float(n)
        if 0 < v < 10000 and v == v:
            return n
    return None

def score_math(question, student_answer, correct_answer):
    student_num = extract_number(student_answer)
    correct_num = extract_number(correct_answer)
    if student_num and correct_num:
        try:
            if abs(float(student_num) - float(correct_num)) < 0.01:
                return 1, f"数值匹配: {student_num} = {correct_num}"
            else:
                return 0, f"数值不匹配: {student_num} ≠ {correct_num}"
        except:
            pass
    # 回退到 LLM 评分
    return None, None

SCORE_PROMPT = """你是公正的评分者。比较学生答案与标准答案。
输出JSON: {"score": 1/0.5/0, "reason":"一句话理由"}
1=核心推理正确, 0.5=方向对但有小错或不完整, 0=核心错误或完全偏离"""

def judge_answer(client, question, student_answer, correct_answer, domain):
    if domain == "math":
        s, r = score_math(question, student_answer, correct_answer)
        if s is not None:
            return {"score": s, "reason": r}
    try:
        r = client.chat.completions.create(
            model="deepseek-reasoner", messages=[
                {"role": "system", "content": SCORE_PROMPT},
                {"role": "user", "content": f"问题: {question}\n标准答案: {correct_answer}\n学生答案: {student_answer[:1200]}"}
            ], temperature=0.1, max_tokens=200, timeout=30
        )
        text = r.choices[0].message.content.strip()
        for p in ["```json","```"]:
            if text.startswith(p): text = text[len(p):]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        return {"score": 0, "reason": f"judge error: {str(e)[:60]}"}


judge_client = get_client("mkeai")
score_client = get_client("deepseek")  # 用DeepSeek-Reasoner评分，更可靠
baselines = [("GPT-4o", "mkeai", "gpt-4o"), ("DeepSeek-Chat", "deepseek", "deepseek-chat")]

print("=" * 60)
print(f"MSCE Phase 0 — {len(BENCHMARK)}题定量验证 v2")
print("=" * 60, flush=True)

msce_scores, gpt4_scores, ds_scores = [], [], []
msce_times = []

for idx, item in enumerate(BENCHMARK):
    q, correct, domain = item["q"], item["answer"], item["domain"]
    print(f"\n[{idx+1}/{len(BENCHMARK)}] [{domain}] {q[:60]}...", flush=True)

    t0 = time.time()
    msce_result = run_msce(q, domain=domain)
    msce_time = time.time() - t0
    msce_times.append(msce_time)

    verdict = msce_result.get("verdict", {})
    top3 = verdict.get("top3", [])
    top1_id = top3[0]["id"] if top3 else "?"

    top1_answer = ""
    for c in msce_result["candidates"]:
        if c["strategy"] == top1_id and c["success"]:
            top1_answer = c["answer"]

    msce_score = judge_answer(score_client, q, top1_answer, correct, domain) if top1_answer else {"score": 0, "reason": "no answer"}
    msce_scores.append(msce_score["score"])

    bl = {}
    for name, ct, mdl in baselines:
        try:
            ans = run_single_model(q, ct, mdl)
            sc = judge_answer(score_client, q, ans, correct, domain)
            bl[name] = sc["score"]
        except Exception as e:
            bl[name] = 0

    gpt4_scores.append(bl["GPT-4o"])
    ds_scores.append(bl["DeepSeek-Chat"])

    surviving = verdict.get("surviving", [])
    eliminated = verdict.get("eliminated", [])
    print(f"  MSCE: {msce_score['score']} (TOP1={top1_id}, elim={len(eliminated)}, surv={len(surviving)}) | GPT-4o: {bl['GPT-4o']} | DeepSeek: {bl['DeepSeek-Chat']} | {msce_time:.0f}s", flush=True)

print(f"\n{'='*60}")
print("汇总统计")
print(f"{'='*60}")
print(f"{'模型':<25} {'平均分':<8} {'正确':<6} {'正确率':<10} {'平均耗时':<10}")
print("-" * 60)
n = len(msce_scores)
print(f"{'MSCE (多模型淘汰赛)':<25} {sum(msce_scores)/n:<8.3f} {sum(1 for s in msce_scores if s==1):<6} {sum(1 for s in msce_scores if s==1)/n*100:<10.1f}% {sum(msce_times)/n:<10.0f}s")
for name, scores in [("GPT-4o", gpt4_scores), ("DeepSeek-Chat", ds_scores)]:
    print(f"{name:<25} {sum(scores)/n:<8.3f} {sum(1 for s in scores if s==1):<6} {sum(1 for s in scores if s==1)/n*100:<10.1f}% {'N/A':<10}")

print(f"\n按领域：")
for domain in ["math", "logic", "science", "verbal"]:
    idxs = [i for i, item in enumerate(BENCHMARK) if item["domain"] == domain]
    if idxs:
        m = sum(msce_scores[i] for i in idxs) / len(idxs)
        g = sum(gpt4_scores[i] for i in idxs) / len(idxs)
        d = sum(ds_scores[i] for i in idxs) / len(idxs)
        print(f"  {domain} ({len(idxs)}题): MSCE={m:.2f}, GPT-4o={g:.2f}, DeepSeek={d:.2f}")

output_path = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.json")
with open(output_path, 'w') as f:
    json.dump({
        "msce_avg": sum(msce_scores)/n,
        "msce_correct": f"{sum(1 for s in msce_scores if s==1)}/{n}",
        "gpt4o_avg": sum(gpt4_scores)/n,
        "gpt4o_correct": f"{sum(1 for s in gpt4_scores if s==1)}/{n}",
        "deepseek_avg": sum(ds_scores)/n,
        "deepseek_correct": f"{sum(1 for s in ds_scores if s==1)}/{n}",
        "avg_time_s": sum(msce_times)/n,
    }, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {output_path}")
