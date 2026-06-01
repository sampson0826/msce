"""MSCE MMLU Baseline — 试点验证"""
import sys, json, time, re
sys.path.insert(0, '/Users/dengxinhang/paper/constraint_residual/msce')
from engine import run_msce, run_single_model, get_client
from datasets import load_dataset

# Sam建议: 选5个代表性学科
PILOT_SUBJECTS = [
    "abstract_algebra",      # STEM 数学
    "college_physics",       # STEM 科学
    "formal_logic",          # 逻辑
    "college_chemistry",     # STEM 化学
    "us_foreign_policy",     # 人文
]

EXTRACT_PROMPT = """从以下答案中提取最终选项字母（A/B/C/D）。
只输出一个字母。如果无法确定，输出 X。
示例输出: B"""

def load_mmlu_questions(subjects, max_per_subject=10):
    """加载MMLU题目"""
    questions = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i, item in enumerate(ds):
            if i >= max_per_subject:
                break
            # 构建问题文本
            choices_text = ""
            letters = ["A", "B", "C", "D"]
            for j, choice in enumerate(item['choices']):
                choices_text += f"{letters[j]}. {choice}\n"

            q_text = f"问题：{item['question']}\n\n选项：\n{choices_text}\n请选出正确答案。只需输出最终答案的字母（A/B/C/D）。"

            questions.append({
                "id": f"{subject}_{i}",
                "subject": subject,
                "q": q_text,
                "answer": letters[item['answer']],
                "answer_text": item['choices'][item['answer']],
            })
    return questions

def extract_letter(text):
    """从答案文本中提取选项字母"""
    if not text:
        return None
    # 1. 找 **X** 格式
    m = re.search(r'\*\*([ABCD])\*\*', text)
    if m: return m.group(1)
    # 2. 找 "答案是 X" 格式
    m = re.search(r'答案[是为：:]\s*([ABCD])', text)
    if m: return m.group(1)
    # 3. 找 "选 X" 格式
    m = re.search(r'[选选择]\s*([ABCD])', text)
    if m: return m.group(1)
    # 4. 末尾找单独字母
    m = re.search(r'\b([ABCD])\b\s*$', text.strip()[-100:])
    if m: return m.group(1)
    # 5. 用LLM提取
    return None

def score_answer(student_answer, correct_letter):
    """评分：提取字母并比对"""
    letter = extract_letter(student_answer)
    if letter:
        return 1 if letter == correct_letter else 0
    # LLM提取fallback
    try:
        client = get_client("deepseek")
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"答案文本：{student_answer[-500:]}"}
            ],
            temperature=0, max_tokens=10, timeout=15
        )
        extracted = r.choices[0].message.content.strip().upper()
        if extracted and extracted[0] in "ABCD":
            return 1 if extracted[0] == correct_letter else 0
    except:
        pass
    return 0

# 域映射（用于自适应裁判）
def map_subject_to_domain(subject):
    math_subjects = {"abstract_algebra", "college_mathematics", "high_school_mathematics", "elementary_mathematics"}
    logic_subjects = {"formal_logic", "logical_fallacies"}
    science_subjects = {"college_physics", "college_chemistry", "college_biology", "high_school_physics", "high_school_chemistry", "electrical_engineering"}
    if subject in math_subjects: return "math"
    if subject in logic_subjects: return "logic"
    if subject in science_subjects: return "science"
    return None

def run_pilot(subjects=None, n_per_subject=10):
    if subjects is None:
        subjects = PILOT_SUBJECTS

    print(f"Loading MMLU questions: {subjects}")
    questions = load_mmlu_questions(subjects, max_per_subject=n_per_subject)
    print(f"Loaded {len(questions)} questions")

    baselines = [("GPT-4o", "mkeai", "gpt-4o"), ("DeepSeek-Chat", "deepseek", "deepseek-chat")]

    msce_scores, gpt4_scores, ds_scores = [], [], []
    msce_times = []

    for idx, item in enumerate(questions):
        print(f"\n[{idx+1}/{len(questions)}] [{item['subject']}] {item['q'][:80]}...", flush=True)

        t0 = time.time()
        domain = map_subject_to_domain(item['subject'])
        msce_result = run_msce(item['q'], domain=domain)
        msce_time = time.time() - t0
        msce_times.append(msce_time)

        verdict = msce_result.get("verdict", {})
        top3 = verdict.get("top3", [])
        top1_id = top3[0]["id"] if top3 else "?"

        top1_answer = ""
        for c in msce_result["candidates"]:
            if c["strategy"] == top1_id and c["success"]:
                top1_answer = c["answer"]

        msce_score = score_answer(top1_answer, item['answer'])
        msce_scores.append(msce_score)

        bl = {}
        for name, ct, mdl in baselines:
            try:
                ans = run_single_model(item['q'], ct, mdl)
                bl[name] = score_answer(ans, item['answer'])
            except Exception as e:
                bl[name] = 0

        gpt4_scores.append(bl["GPT-4o"])
        ds_scores.append(bl["DeepSeek-Chat"])

        surviving = verdict.get("surviving", [])
        eliminated = verdict.get("eliminated", [])
        print(f"  MSCE: {msce_score} (TOP1={top1_id}, elim={len(eliminated)}) | GPT-4o: {bl['GPT-4o']} | DeepSeek: {bl['DeepSeek-Chat']} | {msce_time:.0f}s", flush=True)

    # Summary
    n = len(msce_scores)
    print(f"\n{'='*60}")
    print(f"MMLU Pilot Results — {len(subjects)} subjects, {n} questions")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Avg':<8} {'Correct':<8} {'Rate':<10}")
    print("-"*50)
    print(f"{'MSCE':<25} {sum(msce_scores)/n:<8.3f} {sum(msce_scores):<8} {sum(msce_scores)/n*100:<10.1f}%")
    for name, scores in [("GPT-4o", gpt4_scores), ("DeepSeek-Chat", ds_scores)]:
        print(f"{name:<25} {sum(scores)/n:<8.3f} {sum(scores):<8} {sum(scores)/n*100:<10.1f}%")

    print(f"\nBy subject:")
    for subject in subjects:
        idxs = [i for i, item in enumerate(questions) if item["subject"] == subject]
        if idxs:
            m = sum(msce_scores[i] for i in idxs) / len(idxs)
            g = sum(gpt4_scores[i] for i in idxs) / len(idxs)
            d = sum(ds_scores[i] for i in idxs) / len(idxs)
            print(f"  {subject} ({len(idxs)}): MSCE={m:.2f}, GPT-4o={g:.2f}, DeepSeek={d:.2f}")

    print(f"\nAvg time: {sum(msce_times)/n:.0f}s")

    result_path = os.path.join(os.path.dirname(__file__), "results", "mmlu_pilot_results.json")
    with open(result_path, 'w') as f:
        json.dump({
            "msce_avg": sum(msce_scores)/n,
            "msce_correct": f"{sum(msce_scores)}/{n}",
            "gpt4o_avg": sum(gpt4_scores)/n,
            "deepseek_avg": sum(ds_scores)/n,
            "per_subject": {
                s: {"msce": sum(msce_scores[i] for i in [j for j, item in enumerate(questions) if item["subject"]==s])/len([j for j, item in enumerate(questions) if item["subject"]==s])}
                for s in subjects
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {result_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Questions per subject")
    args = parser.parse_args()

    run_pilot(subjects=PILOT_SUBJECTS, n_per_subject=args.n)
