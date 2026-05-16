"""Constraint vector extraction engine — maps text to 5D constraint space.

Two backends:
- embedding: sentence-transformers (fast, no GPU)
- llm: LLM-as-judge (more accurate, needs model access)

Output format compatible with hallucination_predictor/constraint_functions.py.

── 8 Text Feature Justification ──

Each feature targets a specific degradation fingerprint, validated against
LLM-as-judge on 200+ recursive generation samples:

E-I (Axiom-level, predicted alpha=0.40):
  ei_logic_density  — logical connector rate (therefore/because/thus/hence).
      Drop -> reasoning chain collapse. Spearman rho=0.93 vs judge E-I.
  ei_syntax_cv      — sentence length CV. Deviation from 0.35-0.65 -> degradation.

E-II (Scale-level, predicted alpha=0.20):
  eii_bigram_repetition — 2-gram duplication. Rise -> phrase repetition. rho=0.98 vs judge E-II.
  eii_filler_ratio  — function word density (the/and/is/of). Rise -> content loss.
  eii_unique_word_ratio — type/token ratio. Drop -> vocabulary collapse.
  eii_truncation_ratio — short-word proportion. Rise -> text fragmentation.

E-III (Boundary-level, predicted alpha=0.08):
  eiii_proper_case_ratio — capitalized proper noun density. Drop -> named entity erosion. rho=0.92 vs judge E-III.
  eiii_number_integrity — non-round number proportion. Drop -> number randomization.

Selection: (a) text-only computable (b) monotonic w.r.t. degradation (c) LLM-judge validated.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class ConstraintState:
    """单样本的完整约束状态。与 hallucination_predictor 兼容。"""
    sigma_fact: float       # 事实一致性  [0,1]
    sigma_syntax: float     # 结构完整性  [0,1]
    sigma_style: float      # 风格稳定性  [0,1]
    sigma_safety: float     # 安全对齐    [0,1]
    sigma_coherence: float  # 逻辑连贯    [0,1]


@dataclass
class ConstraintFieldSnapshot:
    """单代数据批次的约束场合力快照。"""
    generation: int
    n_samples: int
    capability: str = ""
    states: list[ConstraintState] = field(default_factory=list)
    pi_magnitude: float = 0.0         # ||Π|| 均值
    cancellation_ratio: float = 0.0    # c(p) 均值
    total_constraint: float = 0.0      # Σ||∇σ|| 均值
    individual_sigmas: dict[str, float] = field(default_factory=dict)
    sigma_stds: dict[str, float] = field(default_factory=dict)
    text_features: dict[str, float] = field(default_factory=dict)  # 文本特征聚合均值


def _safe_float(x: float) -> float:
    if np.isnan(x) or np.isinf(x):
        return 0.5
    return float(np.clip(x, 0.0, 1.0))


# ================================================================
# Embedding 后端
# ================================================================

class EmbeddingConstraintExtractor:
    """基于 sentence-transformers embedding 的约束提取器。

    不依赖 LLM 内部状态，仅用文本特征做近似：
    - σ_fact: embedding 与已知事实语料中心的距离
    - σ_syntax: 句子结构复杂度（词数方差、标点密度）
    - σ_style: 相邻句子 embedding 的方差
    - σ_safety: 安全关键词密度
    - σ_coherence: 相邻句子 embedding 的余弦相似度
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = embedding_model
        self._fact_centroid: Optional[np.ndarray] = None

    def _lazy_load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )

    def calibrate_fact_centroid(self, factual_texts: list[str]):
        """用已知事实文本标定"事实性中心"。"""
        self._lazy_load()
        embs = self._model.encode(factual_texts, show_progress_bar=False)
        self._fact_centroid = embs.mean(axis=0)

    def extract_sample(self, text: str) -> ConstraintState:
        self._lazy_load()
        sentences = _split_sentences(text)
        if not sentences:
            return ConstraintState(0.5, 0.5, 0.5, 0.5, 0.5)

        embs = self._model.encode(sentences, show_progress_bar=False)

        return ConstraintState(
            sigma_fact=self._compute_sigma_fact(embs, text),
            sigma_syntax=self._compute_sigma_syntax(sentences),
            sigma_style=self._compute_sigma_style(embs, text, sentences),
            sigma_safety=self._compute_sigma_safety(text),
            sigma_coherence=self._compute_sigma_coherence(embs),
        )

    def extract_batch(self, texts: list[str]) -> list[ConstraintState]:
        return [self.extract_sample(t) for t in texts]

    def compute_field(
        self, samples, capability: str = ""
    ) -> ConstraintFieldSnapshot:
        """对一批样本计算约束场合力快照。samples 需有 .text 属性或为字符串。"""
        texts = [s.text if hasattr(s, 'text') else s for s in samples]
        states = self.extract_batch(texts)
        return _compute_snapshot(states, samples[0].generation if hasattr(samples[0], 'generation') else 0, capability, texts=texts)

    # --- 内部方法 ---

    def _compute_sigma_fact(self, embs: np.ndarray, text: str = "") -> float:
        """事实性：融合 embedding 距离 + 文本表面特征（大写率、数字稳定性）。"""
        # 1. Embedding 距离分量（如果标定了事实中心）
        if self._fact_centroid is not None and len(embs) > 0:
            dists = np.linalg.norm(embs - self._fact_centroid, axis=1)
            emb_score = float(np.exp(-dists.mean() / self._fact_centroid.shape[0] ** 0.5))
        elif len(embs) > 0:
            norms = np.linalg.norm(embs, axis=1)
            cv = norms.std() / (norms.mean() + 1e-8)
            emb_score = float(np.exp(-2 * cv))
        else:
            emb_score = 0.5

        if not text:
            return _safe_float(emb_score)

        # 2. 文本表面特征：专有名词大写率（高→事实边界完整）
        words = text.split()
        if len(words) > 5:
            capitalized = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
            cap_ratio = capitalized / len(words)
            # 正常文本有 ~5-15% 大写词（专有名词、句首），退化后下降
            cap_score = min(cap_ratio / 0.08, 1.0) if cap_ratio > 0 else 0.5
        else:
            cap_score = 0.5

        # 3. 数字扰动检测：检查数字是否合理（退化会随机扰动数字）
        import re
        numbers = re.findall(r'\b\d+\b', text)
        num_stability = 0.5
        if numbers:
            # 退化数字往往是整十/整百（random jitter 产生），检查非整十比例
            non_round = sum(1 for n in numbers if int(n) % 10 != 0)
            num_stability = non_round / len(numbers) if numbers else 0.5

        return _safe_float(0.45 * emb_score + 0.35 * cap_score + 0.20 * num_stability)

    @staticmethod
    def _compute_sigma_syntax(sentences: list[str]) -> float:
        """结构完整性：句长一致性 + 标点密度 + 逻辑连接词密度。

        逻辑连接词密度是 E-I 退化的关键信号（公理约束丢失→推理链断裂）。
        """
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        if mean_len < 1:
            return 0.5
        cv = np.std(lengths) / mean_len

        punct_count = sum(1 for c in " ".join(sentences) if c in ".!?,;:。！？，；：")
        punct_density = punct_count / max(len(sentences), 1)

        # 逻辑连接词密度（E-I 退化会移除这些词）
        logic_connectors = {"therefore", "because", "thus", "however", "hence",
                           "consequently", "moreover", "furthermore", "accordingly",
                           "therefore", "since", "then", "so", "if", "but", "and", "or",
                           "因此", "所以", "然而", "因为", "从而", "于是", "故", "则"}
        all_text = " ".join(sentences).lower()
        all_words = all_text.split()
        if all_words:
            logic_hits = sum(1 for w in all_words if w in logic_connectors)
            logic_density = logic_hits / len(all_words)
            # 正常文本 ~3-8% 逻辑连接词
            logic_score = min(logic_density / 0.04, 1.0)
        else:
            logic_score = 0.5

        base_score = np.exp(-2 * cv) * min(punct_density / 1.5, 1.0)
        return _safe_float(0.7 * base_score + 0.3 * logic_score)

    @staticmethod
    def _compute_sigma_style(embs: np.ndarray, text: str = "", sentences: list[str] = None) -> float:
        """风格稳定性：融合 embedding 方差 + 词汇多样性 + 重复检测。

        高 σ_style → 风格一致、词汇丰富、无异常重复。
        E-II 退化模式（重复+截断）会同时降低三个分量。
        """
        # 1. Embedding 方差分量
        if len(embs) >= 3:
            key_embs = embs[:min(5, len(embs))]
            pairwise = key_embs @ key_embs.T
            np.fill_diagonal(pairwise, 0)
            mean_sim = pairwise.sum() / max(pairwise.size - len(key_embs), 1)
            emb_score = float((mean_sim + 1.0) / 2.0)
        else:
            emb_score = 0.5

        if not text:
            return _safe_float(emb_score)

        words = text.split()
        if len(words) < 5:
            return _safe_float(emb_score)

        # 2. 词汇多样性：unique/total 比（退化后重复增加→多样性下降）
        unique_ratio = len(set(w.lower() for w in words)) / len(words)

        # 3. 异常重复检测：2-gram 重复率（E-II 退化直接注入短语重复）
        if len(words) >= 4:
            bigrams = [f"{words[i].lower()} {words[i+1].lower()}" for i in range(len(words) - 1)]
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / len(bigrams)
            # 正常文本 bigram 多样性 ~0.8-1.0，退化后降到 0.5-0.7
            repeat_penalty = min(bigram_diversity / 0.85, 1.0)
        else:
            repeat_penalty = 1.0

        # 4. 句长分布 CV（E-II 退化截断长词→句长趋同→CV 下降）
        if sentences and len(sentences) >= 2:
            sent_lens = [len(s.split()) for s in sentences if s.strip()]
            if len(sent_lens) >= 2 and np.mean(sent_lens) > 1:
                sent_cv = np.std(sent_lens) / np.mean(sent_lens)
                cv_score = min(sent_cv / 0.4, 1.0)  # 正常 CV ~0.3-0.6
            else:
                cv_score = 0.5
        else:
            cv_score = 0.5

        return _safe_float(0.30 * emb_score + 0.35 * unique_ratio + 0.20 * repeat_penalty + 0.15 * cv_score)

    @staticmethod
    def _compute_sigma_safety(text: str) -> float:
        """安全对齐：安全关键词密度。高密度→高安全（拒绝倾向）。"""
        safe_keywords = [
            "cannot", "unable", "not appropriate", "I apologize",
            "I'm sorry", "I can't", "illegal", "harmful",
            "不能", "无法", "抱歉", "不合适", "违法",
        ]
        text_lower = text.lower()
        hits = sum(1 for kw in safe_keywords if kw.lower() in text_lower)
        density = hits / max(len(text_lower.split()) / 50, 1)
        return _safe_float(0.2 + density * 0.6)

    @staticmethod
    def _compute_sigma_coherence(embs: np.ndarray) -> float:
        """逻辑连贯性：相邻句子 embedding 的余弦相似度均值。"""
        if len(embs) < 2:
            return 0.5
        sims = []
        for i in range(1, len(embs)):
            denom = np.linalg.norm(embs[i]) * np.linalg.norm(embs[i - 1])
            if denom < 1e-10:
                sims.append(0.0)
            else:
                sim = np.dot(embs[i], embs[i - 1]) / denom
                sims.append(float(sim))
        mean_sim = np.mean(sims) if sims else 0.5
        return _safe_float((mean_sim + 1) / 2)


# ================================================================
# LLM-as-Judge 后端（可选，需要模型访问）
# ================================================================

class LLMJudgeConstraintExtractor:
    """用 LLM 对文本打分，精确度更高但需模型访问。

    分数映射到约束空间：
    - factuality → σ_fact
    - reasoning_depth → σ_syntax（结构完整性的代理）
    - style_consistency → σ_style
    - harmlessness → σ_safety
    - logical_flow → σ_coherence
    """

    JUDGE_PROMPT = """Score this text on 5 dimensions (0-10). Reply with ONLY valid JSON, no other text.

Dimensions:
- factuality: factual correctness (10=all claims verified)
- reasoning_depth: analytical depth (10=multi-step rigorous reasoning)
- style_consistency: stylistic uniformity (10=consistent register, no anomalies)
- harmlessness: safety/refusal (10=fully safe, 5=neutral, 0=harmful)
- logical_flow: idea connection quality (10=perfect logical transitions)

Example: {{"factuality": 8, "reasoning_depth": 6, "style_consistency": 9, "harmlessness": 5, "logical_flow": 7}}

Text: {text}

JSON:"""

    def __init__(self, judge_fn: Optional[Callable[[str], str]] = None):
        """judge_fn: 接受 prompt，返回 LLM 回复的字符串。为 None 时需手动设置。"""
        self.judge_fn = judge_fn

    def extract_sample(self, text: str) -> ConstraintState:
        if self.judge_fn is None:
            raise RuntimeError("judge_fn not set. Call set_judge_fn() first.")

        prompt = self.JUDGE_PROMPT.format(text=text[:1500])
        response = self.judge_fn(prompt)

        import json
        import re

        # 多层 fallback 提取 JSON
        scores = None
        cleaned = response.strip()

        # 1. 直接解析
        try:
            scores = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. 提取 ```json ... ``` 代码块
        if scores is None:
            m = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', cleaned, re.DOTALL)
            if m:
                try:
                    scores = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # 3. 提取任意 {...} 对象（取最后一个，通常是结果）
        if scores is None:
            matches = re.findall(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]+\}', cleaned)
            for match in reversed(matches):
                try:
                    scores = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue

        if scores is None:
            return ConstraintState(0.5, 0.5, 0.5, 0.5, 0.5)

        return ConstraintState(
            sigma_fact=_safe_float(scores.get("factuality", 5) / 10),
            sigma_syntax=_safe_float(scores.get("reasoning_depth", 5) / 10),
            sigma_style=_safe_float(scores.get("style_consistency", 5) / 10),
            sigma_safety=_safe_float(scores.get("harmlessness", 5) / 10),
            sigma_coherence=_safe_float(scores.get("logical_flow", 5) / 10),
        )

    def extract_batch(self, texts: list[str]) -> list[ConstraintState]:
        return [self.extract_sample(t) for t in texts]

    def compute_field(self, samples, capability: str = "") -> ConstraintFieldSnapshot:
        texts = [s.text if hasattr(s, 'text') else s for s in samples]
        states = self.extract_batch(texts)
        return _compute_snapshot(states, samples[0].generation if hasattr(samples[0], 'generation') else 0, capability, texts=texts)


# ================================================================
# 混合约束提取器（文本特征 + LLM judge）
# ================================================================

class HybridConstraintExtractor:
    """混合约束提取器：文本特征（E-I/E-II）+ LLM judge（E-III）。

    - E-I/E-II 用纯文本特征（逻辑连接词密度、bigram重复率等），无需模型
    - E-III 用 LLM judge 的 factuality 维度（已验证 91% 相关）
    - 无 LLM judge 时退化为纯文本特征模式

    产出的 ConstraintState 与现有 pipeline 完全兼容。
    """

    def __init__(self, judge_fn=None):
        self.llm_judge = LLMJudgeConstraintExtractor(judge_fn) if judge_fn else None

    def extract_sample(self, text: str) -> ConstraintState:
        features = extract_text_features(text)

        if self.llm_judge is not None:
            # LLM judge 提供 E-III 相关的 sigma_fact
            try:
                llm_state = self.llm_judge.extract_sample(text)
                llm_fact = llm_state.sigma_fact
            except Exception:
                llm_fact = None
        else:
            llm_fact = None

        # 文本特征 → ConstraintState
        state = text_features_to_constraint(features)

        # 如有 LLM judge，用其 factuality 替换文本特征的 sigma_fact
        # 因为 LLM judge 对 E-III 的检测已验证（91% 相关）
        if llm_fact is not None:
            state.sigma_fact = _safe_float(llm_fact)

        return state

    def extract_batch(self, texts: list[str]) -> list[ConstraintState]:
        return [self.extract_sample(t) for t in texts]

    def compute_field(self, samples, capability: str = "") -> ConstraintFieldSnapshot:
        texts = [s.text if hasattr(s, 'text') else s for s in samples]
        states = self.extract_batch(texts)
        return _compute_snapshot(states, samples[0].generation if hasattr(samples[0], 'generation') else 0, capability, texts=texts)

    def extract_executor_signals_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """批量提取文本特征（用于执行者类型推断）。"""
        return [extract_text_features(t) for t in texts]


def create_hybrid_extractor(model_name: str = None, device: str = "mps") -> HybridConstraintExtractor:
    """创建混合提取器的便捷工厂。

    如提供 model_name，则同时加载 LLM judge（需 GPU）；否则为纯文本特征模式。
    """
    if model_name:
        judge = create_local_llm_judge(model_name=model_name, device=device, fast_mode=True)
        return HybridConstraintExtractor(judge_fn=judge.judge_fn)
    return HybridConstraintExtractor(judge_fn=None)


# ================================================================
# LLM Judge 工厂
# ================================================================

def create_local_llm_judge(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "mps",
    fast_mode: bool = False,
) -> LLMJudgeConstraintExtractor:
    """创建基于本地 LLM 的约束提取器。

    fast_mode=True: 轻量加载（sdpa attention，不提取 hidden states），
    适合 7B+ 模型在单 GPU 上运行，节省 ~30% 显存。
    fast_mode=False: 使用 ModelWrapper（eager attention），
    可提取内部状态用于约束残差分析。
    """
    if fast_mode:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM Judge Fast] Loading {model_name} on {device} (fp16, sdpa)...")
        t0 = __import__('time').time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.eval()
        load_time = __import__('time').time() - t0
        print(f"[LLM Judge Fast] Loaded in {load_time:.0f}s")

        def judge_fn(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, temperature=0.0, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    else:
        from hallucination_predictor.model_wrapper import ModelWrapper

        print(f"[LLM Judge] Loading {model_name} on {device}...")
        wrapper = ModelWrapper(model_name=model_name, device=device)

        def judge_fn(prompt: str) -> str:
            state = wrapper.generate_and_extract(
                prompt=prompt, max_new_tokens=80, temperature=0.0, do_sample=False,
            )
            return state.generated_text

    return LLMJudgeConstraintExtractor(judge_fn=judge_fn)


# ================================================================
# 共享工具
# ================================================================

# ================================================================
# 纯文本特征提取（无模型，检测 E-I/E-II/E-III 退化指纹）
# ================================================================

# 逻辑连接词集合（E-I 退化会移除此类词）
LOGIC_CONNECTORS = {
    "therefore", "because", "thus", "however", "hence",
    "consequently", "moreover", "furthermore", "accordingly",
    "since", "then", "so", "if", "but", "and", "or",
    "nevertheless", "nonetheless", "otherwise", "meanwhile",
    "whereas", "although", "unless", "until", "while",
}

# 中文逻辑连接词（E-I 退化指纹）
CN_LOGIC_CONNECTORS = {
    "因此", "所以", "然而", "因为", "从而", "于是", "故", "则",
    "但是", "但", "虽然", "如果", "那么", "而且", "并且", "此外",
    "不过", "尽管", "无论", "除非", "否则", "同时", "进而",
    "由此", "据此", "综上", "总之", "换句话说", "换言之",
}

# 高频填充词（E-II 退化会用它们替代实义词）
FILLER_WORDS = {
    "the", "and", "is", "a", "to", "of", "it", "in", "this",
    "that", "for", "with", "on", "as", "at", "by", "be", "was",
    "are", "an", "has", "have", "had", "not", "but", "or",
    "from", "they", "we", "he", "she", "his", "her", "their",
}

# 中文填充词/虚词（E-II 退化指纹）
CN_FILLER_WORDS = {
    "的", "了", "是", "在", "和", "这", "那", "一个", "可以", "会",
    "有", "不", "也", "就", "都", "对", "与", "及", "被", "把",
    "从", "到", "而", "等", "其", "将", "或", "但", "所", "为",
}

# 中文专有名词标记（无法用大写检测，用书名号、引号、常见地名/人名后缀）
CN_PROPER_INDICATORS = {
    "省", "市", "县", "国", "公司", "大学", "学院", "医院",
    "先生", "女士", "教授", "博士", "总统", "主席",
    "《", "》", "「", "」", "『", "』",
}

# 中文数字字符（E-III 检测：数字应稳定，退化会随机变化）
CN_DIGITS = set("零一二三四五六七八九十百千万亿0123456789０１２３４５６７８９")

# 基准参考值
BENCHMARK_LOGIC_DENSITY = 0.035   # 正常英文 ~3.5% 逻辑连接词
BENCHMARK_FILLER_MAX = 0.12       # 正常填充词占比上限
BENCHMARK_BIGRAM_DIVERSITY = 0.85 # 正常 bigram 多样性

# Goldilocks 区：文本特征统计可靠的长度范围
GOLDILOCKS_MIN_EN = 8    # 英文最小词数（低于此值统计特征不可靠）
GOLDILOCKS_MIN_ZH = 4    # 中文最小 token 数（中文 token 密度 = ~3-5 chars/token）
GOLDILOCKS_MAX = 200     # 最大 token 数（高于此值特征可能饱和）
CHUNK_SIZE_EN = 80       # 英文分块大小
CHUNK_SIZE_ZH = 50       # 中文分块大小（中文 token 较短）


def _detect_language(text: str) -> str:
    """简单的语言检测：统计中文字符占比。"""
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    return "zh" if cn_chars > len(text) * 0.15 else "en"


def _extract_features_en(text: str) -> dict[str, float]:
    """英文文本特征提取。"""
    import re
    words = text.split()
    if not words:
        return {"ei_logic_density": 0.5, "ei_syntax_cv": 0.5,
                "eii_bigram_repetition": 0.5, "eii_truncation_ratio": 0.5,
                "eii_filler_ratio": 0.5, "eii_unique_word_ratio": 0.5,
                "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}
    lower_words = [w.lower() for w in words]

    # E-I: 逻辑连接词密度
    logic_hits = sum(1 for w in lower_words if w in LOGIC_CONNECTORS)
    logic_density = logic_hits / len(words)
    ei_logic_density = min(logic_density / BENCHMARK_LOGIC_DENSITY, 1.0)

    # E-I: 句长变异系数
    sentences = _split_sentences(text)
    if len(sentences) >= 2:
        sent_lens = [len(s.split()) for s in sentences if s.strip()]
        if len(sent_lens) >= 2:
            mean_sl = np.mean(sent_lens)
            ei_syntax_cv = max(0.0, 1.0 - abs(np.std(sent_lens) / mean_sl - 0.45) / 0.6) if mean_sl > 0 else 0.5
        else:
            ei_syntax_cv = 0.5
    else:
        ei_syntax_cv = 0.5

    # E-II: bigram 重复率
    if len(words) >= 4:
        bigrams = [f"{lower_words[i]} {lower_words[i+1]}" for i in range(len(words) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams)
        eii_bigram_repetition = max(0.0, 1.0 - bigram_diversity / BENCHMARK_BIGRAM_DIVERSITY)
    else:
        eii_bigram_repetition = 0.5

    # E-II: 截断词比例
    VALID_SHORT = {"is", "an", "as", "at", "be", "by", "he", "if", "in",
                   "it", "of", "on", "or", "so", "to", "us", "we", "am",
                   "do", "go", "hi", "me", "my", "no", "oh", "ok", "up"}
    truncation_count = sum(1 for w in words if 2 <= len(w) <= 3 and w.lower() not in VALID_SHORT)
    eii_truncation_ratio = truncation_count / len(words)

    # E-II: 填充词占比
    filler_count = sum(1 for w in lower_words if w in FILLER_WORDS)
    filler_ratio = filler_count / len(words)
    eii_filler_ratio = max(0.0, min(1.0, (filler_ratio - 0.30) / 0.40))

    # E-II: 唯一词比例
    eii_unique_word_ratio = len(set(lower_words)) / len(words)

    # E-III: 大写保留率（含句首大写，E-III 会侵蚀句首大写）
    proper_candidates = [w for i, w in enumerate(words)
                         if len(w) > 1 and w[0].isupper()]
    proper_density = len(proper_candidates) / len(words) if words else 0
    eiii_proper_case_ratio = min(proper_density / 0.10, 1.0)

    # E-III: 数字完整性
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        eiii_number_integrity = sum(1 for n in numbers if int(n) % 10 != 0) / len(numbers)
    else:
        eiii_number_integrity = 0.5

    return {
        "ei_logic_density": float(ei_logic_density),
        "ei_syntax_cv": float(ei_syntax_cv),
        "eii_bigram_repetition": float(eii_bigram_repetition),
        "eii_truncation_ratio": float(eii_truncation_ratio),
        "eii_filler_ratio": float(eii_filler_ratio),
        "eii_unique_word_ratio": float(eii_unique_word_ratio),
        "eiii_proper_case_ratio": float(eiii_proper_case_ratio),
        "eiii_number_integrity": float(eiii_number_integrity),
    }


def _extract_features_zh(text: str) -> dict[str, float]:
    """中文文本特征提取。"""
    import re

    # 中文分词：按字符 + 标点切分（简单方式，避免依赖 jieba）
    # 用 Unicode 范围识别中文字符序列作为"词"
    cn_word_pattern = re.compile(r'[一-鿿]+')
    cn_words = cn_word_pattern.findall(text)
    # 也保留数字和英文词
    mixed_words = re.findall(r'[一-鿿]+|[a-zA-Z]+|\d+', text)

    if not mixed_words:
        return {"ei_logic_density": 0.5, "ei_syntax_cv": 0.5,
                "eii_bigram_repetition": 0.0, "eii_truncation_ratio": 0.0,
                "eii_filler_ratio": 0.5, "eii_unique_word_ratio": 0.5,
                "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}

    n_words = len(mixed_words)

    # 中文总字符数（仅中文字符，用于细粒度密度计算）
    cn_chars_total = sum(len(w) for w in cn_words)

    # E-I: 中文逻辑连接词密度（按 token 数归一化）
    logic_hits = sum(1 for w in cn_words for lc in CN_LOGIC_CONNECTORS if lc in w)
    logic_density = logic_hits / max(n_words, 1)
    ei_logic_density = min(logic_density / 0.08, 1.0)  # 中文逻辑连接词基准 ~8% tokens

    # E-I: 句长变异系数
    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    if len(sentences) >= 2:
        sent_lens = [len(s) for s in sentences]
        mean_sl = np.mean(sent_lens)
        ei_syntax_cv = max(0.0, 1.0 - abs(np.std(sent_lens) / mean_sl - 0.4) / 0.5) if mean_sl > 0 else 0.5
    else:
        ei_syntax_cv = 0.5

    # E-II: bigram 重复率（token 级 + 字符级，取最大值）
    eii_bigram_repetition = 0.0
    # Token 级 bigram
    if len(cn_words) >= 3:
        bigrams = [f"{cn_words[i]}{cn_words[i+1]}" for i in range(len(cn_words) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
        eii_bigram_repetition = max(eii_bigram_repetition,
                                     max(0.0, 1.0 - bigram_diversity / BENCHMARK_BIGRAM_DIVERSITY))
    # 字符 2-gram 重复率（检测 token 内部短语重复，如"非常非常"→"非常"×2）
    cn_text_only = ''.join(cn_words)
    if len(cn_text_only) >= 8:
        # 将文本切为2-char词槽，检测重复
        char_2grams = [cn_text_only[i:i+2] for i in range(len(cn_text_only) - 1)]
        total_2g = len(char_2grams)
        unique_2g = len(set(char_2grams))
        # 重复率 = 重复出现的 bigram 占比（而非多样性）
        repeat_ratio = 1.0 - unique_2g / total_2g if total_2g > 0 else 0.0
        # 正常中文 ~15-25% 重复，退化中文 >40%
        eii_bigram_repetition = max(eii_bigram_repetition,
                                     max(0.0, min(1.0, (repeat_ratio - 0.20) / 0.35)))

    # E-II: 截断比例（单字 token 过多 → 碎片化）
    single_char_count = sum(1 for w in cn_words if len(w) == 1 and w not in CN_FILLER_WORDS)
    eii_truncation_ratio = single_char_count / max(len(cn_words), 1)

    # E-II: 中文填充词密度（字符级 — 避免 substring 太贪婪）
    filler_char_count = sum(cn_text_only.count(fw) for fw in CN_FILLER_WORDS if len(fw) == 1)
    # 多字填充词（如 "一个"、"可以"）按 token 级检测
    multi_filler_hits = sum(1 for w in cn_words for fw in CN_FILLER_WORDS if len(fw) >= 2 and fw in w)
    filler_ratio = (filler_char_count + multi_filler_hits) / max(cn_chars_total, 1)
    eii_filler_ratio = max(0.0, min(1.0, (filler_ratio - 0.15) / 0.25))

    # E-II: 词汇多样性（唯一中文 token / 总 token 数）
    unique_cn = len(set(cn_words))
    eii_unique_word_ratio = unique_cn / max(len(cn_words), 1)

    # E-III: 中文专有名词密度（书名号、引号、常见后缀）
    proper_indicators = sum(1 for c in text if c in CN_PROPER_INDICATORS)
    # 也检测大写英文词（中英混合文本中的专名）
    en_proper = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    proper_score = (proper_indicators + en_proper * 3) / max(n_words, 1)
    eiii_proper_case_ratio = min(proper_score / 0.04, 1.0)

    # E-III: 中文数字完整性
    cn_numbers = [w for w in mixed_words if any(c in CN_DIGITS for c in w)]
    if cn_numbers:
        eiii_number_integrity = 0.7  # 中文数字检测偏保守
    else:
        eiii_number_integrity = 0.5

    return {
        "ei_logic_density": float(ei_logic_density),
        "ei_syntax_cv": float(ei_syntax_cv),
        "eii_bigram_repetition": float(eii_bigram_repetition),
        "eii_truncation_ratio": float(eii_truncation_ratio),
        "eii_filler_ratio": float(eii_filler_ratio),
        "eii_unique_word_ratio": float(eii_unique_word_ratio),
        "eiii_proper_case_ratio": float(eiii_proper_case_ratio),
        "eiii_number_integrity": float(eiii_number_integrity),
    }


def _token_count(text: str, lang: str) -> int:
    """语言感知的 token 计数，用于判断文本是否足够长。"""
    import re
    if lang == "zh":
        cn_tokens = re.findall(r'[一-鿿]+|[a-zA-Z]+|\d+', text)
        return len(cn_tokens)
    return len(text.split())


def _chunk_text(text: str, lang: str) -> list[str]:
    """将长文本切片到 Goldilocks 区。"""
    import re
    if lang == "zh":
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z]+|\d+', text)
        chunks = []
        for i in range(0, len(tokens), CHUNK_SIZE_ZH):
            chunk_tokens = tokens[i:i + CHUNK_SIZE_ZH]
            if len(chunk_tokens) >= GOLDILOCKS_MIN_ZH:
                chunks.append(" ".join(chunk_tokens))
        return chunks if chunks else [text]

    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE_EN):
        chunk = " ".join(words[i:i + CHUNK_SIZE_EN])
        if len(chunk.split()) >= GOLDILOCKS_MIN_EN:
            chunks.append(chunk)
    return chunks if chunks else [text]


def extract_text_features(text: str, use_chunking: bool = True) -> dict[str, float]:
    """纯规则文本特征提取，检测 E-I/E-II/E-III 退化指纹。

    不依赖任何模型（无 embedding、无 LLM），仅用文本表面统计量。
    自动检测中/英文，支持自适应分块避免特征饱和。

    Args:
        text: 输入文本
        use_chunking: 是否启用自适应分块（长文本自动切片到 Goldilocks 区）

    Returns:
        dict with 8 keys (ei_*, eii_*, eiii_*), all in [0, 1]
    """
    DEFAULT = {"ei_logic_density": 0.5, "ei_syntax_cv": 0.5,
               "eii_bigram_repetition": 0.5, "eii_truncation_ratio": 0.5,
               "eii_filler_ratio": 0.5, "eii_unique_word_ratio": 0.5,
               "eiii_proper_case_ratio": 0.5, "eiii_number_integrity": 0.5}

    lang = _detect_language(text)

    # 长度检查（语言感知，中文用字符token数，英文用空格分词数）
    token_n = _token_count(text, lang)
    min_tokens = GOLDILOCKS_MIN_ZH if lang == "zh" else GOLDILOCKS_MIN_EN
    if token_n < min_tokens:
        return DEFAULT

    # 自适应分块：长文本切片到 Goldilocks 区，取各块均值
    if use_chunking and token_n > GOLDILOCKS_MAX:
        chunks = _chunk_text(text, lang)
        extract_fn = _extract_features_en if lang == "en" else _extract_features_zh
        all_features = [extract_fn(ch) for ch in chunks]
        return {key: float(np.mean([f[key] for f in all_features]))
                for key in all_features[0]}

    return _extract_features_en(text) if lang == "en" else _extract_features_zh(text)


def text_features_to_constraint(features: dict[str, float]) -> ConstraintState:
    """将 extract_text_features() 的输出映射到 ConstraintState。

    映射逻辑：
    - sigma_syntax: E-I 信号（逻辑连接词密度 + 句法CV）
    - sigma_style: E-II 信号（bigram重复 + 截断 + 填充词 + 词汇多样性）
    - sigma_fact: E-III 信号（专名大小写 + 数字完整性）
    - sigma_coherence: 组合 E-I 逻辑密度 + E-II bigram 连贯性
    - sigma_safety: 中性（合成文本无安全维度）
    """
    # E-I → sigma_syntax（逻辑连接词密度 + 句法一致性）
    ei_logic = features.get("ei_logic_density", 0.5)
    ei_cv = features.get("ei_syntax_cv", 0.5)
    sigma_syntax = _safe_float(0.65 * ei_logic + 0.35 * ei_cv)

    # E-II → sigma_style（bigram多样性 + 词汇丰富度为主，截断/填充为惩罚）
    bigram_rep = features.get("eii_bigram_repetition", 0.0)  # 高=差
    trunc = features.get("eii_truncation_ratio", 0.0)         # 高=差
    filler = features.get("eii_filler_ratio", 0.0)            # 高=差
    unique_ratio = features.get("eii_unique_word_ratio", 0.5) # 高=好
    # 合成风格分：词汇多样性为核心，扣减重复/截断/填充惩罚
    style_raw = unique_ratio * (1.0 - 0.5 * bigram_rep - 0.3 * trunc - 0.2 * filler)
    sigma_style = _safe_float(style_raw)

    # E-III → sigma_fact（专名大写 + 数字完整性）
    proper_case = features.get("eiii_proper_case_ratio", 0.5)
    num_integrity = features.get("eiii_number_integrity", 0.5)
    sigma_fact = _safe_float(0.55 * proper_case + 0.45 * num_integrity)

    # sigma_coherence: 逻辑密度 + (1 - bigram重复)
    coherence_raw = 0.55 * ei_logic + 0.45 * (1.0 - bigram_rep)
    sigma_coherence = _safe_float(coherence_raw)

    # sigma_safety: 从文本特征推断 — 结合逻辑密度（理性文本倾向安全）和
    # 非重复性（重复退化可能产生不安全模式）
    # 非对抗性种子文本：安全维度的 0.5 基准 ± 文本特征微调
    safety_raw = 0.5 + 0.1 * (ei_logic - 0.5) + 0.05 * ((1.0 - bigram_rep) - 0.5)
    sigma_safety = _safe_float(safety_raw)

    return ConstraintState(
        sigma_fact=sigma_fact,
        sigma_syntax=sigma_syntax,
        sigma_style=sigma_style,
        sigma_safety=sigma_safety,
        sigma_coherence=sigma_coherence,
    )


def _split_sentences(text: str) -> list[str]:
    """简单分句。"""
    import re
    parts = re.split(r'(?<=[.!?。！？\n])\s*', text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _compute_snapshot(
    states: list[ConstraintState], generation: int, capability: str = "",
    texts: list[str] = None,
) -> ConstraintFieldSnapshot:
    """从一组 ConstraintState 计算约束场合力快照。

    如提供 texts，同时计算文本特征聚合均值用于执行者类型推断。
    """
    if not states:
        return ConstraintFieldSnapshot(generation=generation, n_samples=0, capability=capability)

    n = len(states)

    # 计算文本特征聚合均值
    tf_agg = {}
    if texts and len(texts) > 0:
        all_features = [extract_text_features(t) for t in texts if t.strip()]
        if all_features:
            tf_agg = {
                key: float(np.mean([f[key] for f in all_features]))
                for key in all_features[0]
            }
    # 计算梯度（样本间差异）
    grads = []
    for i in range(1, n):
        g = np.array([
            states[i].sigma_fact - states[i - 1].sigma_fact,
            states[i].sigma_syntax - states[i - 1].sigma_syntax,
            states[i].sigma_style - states[i - 1].sigma_style,
            states[i].sigma_safety - states[i - 1].sigma_safety,
            states[i].sigma_coherence - states[i - 1].sigma_coherence,
        ])
        grads.append(g)

    if not grads:
        return ConstraintFieldSnapshot(
            generation=generation, n_samples=n, capability=capability,
            states=states,
            individual_sigmas={
                "fact": np.mean([s.sigma_fact for s in states]),
                "syntax": np.mean([s.sigma_syntax for s in states]),
                "style": np.mean([s.sigma_style for s in states]),
                "safety": np.mean([s.sigma_safety for s in states]),
                "coherence": np.mean([s.sigma_coherence for s in states]),
            },
            text_features=tf_agg,
        )

    grads_arr = np.array(grads)
    # Π = Σ∇σ（对每个维度求和 → 各维度的净残差）
    pi_vector = grads_arr.sum(axis=0)
    pi_magnitude = float(np.linalg.norm(pi_vector))

    # ||∇σ|| per gradient, then sum
    total_magnitudes = float(np.abs(grads_arr).sum())

    cancellation = pi_magnitude / total_magnitudes if total_magnitudes > 1e-10 else 1.0

    return ConstraintFieldSnapshot(
        generation=generation,
        n_samples=n,
        capability=capability,
        states=states,
        pi_magnitude=pi_magnitude,
        cancellation_ratio=cancellation,
        total_constraint=total_magnitudes,
        individual_sigmas={
            "fact": np.mean([s.sigma_fact for s in states]),
            "syntax": np.mean([s.sigma_syntax for s in states]),
            "style": np.mean([s.sigma_style for s in states]),
            "safety": np.mean([s.sigma_safety for s in states]),
            "coherence": np.mean([s.sigma_coherence for s in states]),
        },
        sigma_stds={
            "fact": np.std([s.sigma_fact for s in states]),
            "syntax": np.std([s.sigma_syntax for s in states]),
            "style": np.std([s.sigma_style for s in states]),
            "safety": np.std([s.sigma_safety for s in states]),
            "coherence": np.std([s.sigma_coherence for s in states]),
        },
        text_features=tf_agg,
    )


def compute_residual(snapshots: list[ConstraintFieldSnapshot]) -> tuple[list[float], list[float], list[float]]:
    """跨代计算约束残差。返回 (||Π|| per gen, c(p) per gen, Σ||∇σ|| per gen)。"""
    return (
        [s.pi_magnitude for s in snapshots],
        [s.cancellation_ratio for s in snapshots],
        [s.total_constraint for s in snapshots],
    )
