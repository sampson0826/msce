"""
数据代际血统解析 — 标记每个样本的合成数据代数和来源。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# 预定义能力维度标签
CAPABILITY_TAGS = [
    "math_reasoning",
    "code_generation",
    "factual_knowledge",
    "logical_consistency",
    "style_diversity",
    "safety_alignment",
    "instruction_following",
    "creative_writing",
    "translation",
    "summarization",
]


@dataclass
class DataSample:
    text: str
    generation: int = 0
    source_model: str = "human"
    capability_tags: list[str] = field(default_factory=list)
    sample_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DatasetLineage:
    samples: list[DataSample]
    generations: dict[int, list[DataSample]] = field(default_factory=dict)
    capability_coverage: dict[str, dict[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generations:
            self._index()

    def _index(self):
        self.generations = {}
        self.capability_coverage = {}
        for s in self.samples:
            self.generations.setdefault(s.generation, []).append(s)
            for tag in s.capability_tags:
                cov = self.capability_coverage.setdefault(tag, {})
                cov[s.generation] = cov.get(s.generation, 0) + 1

    @property
    def n_generations(self) -> int:
        return max(self.generations.keys()) + 1 if self.generations else 0

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def generation_summary(self) -> dict:
        return {
            gen: {
                "n_samples": len(samples),
                "models": list(set(s.source_model for s in samples)),
                "capability_tags": list(set(
                    tag for s in samples for tag in s.capability_tags
                )),
            }
            for gen, samples in sorted(self.generations.items())
        }

    def samples_by_capability(self, capability: str, generation: int) -> list[DataSample]:
        return [
            s for s in self.generations.get(generation, [])
            if capability in s.capability_tags
        ]

    def save(self, path: str):
        """Save lineage to JSONL file."""
        import json
        with open(path, "w") as f:
            for s in self.samples:
                obj = {
                    "text": s.text,
                    "generation": s.generation,
                    "source_model": s.source_model,
                    "capability_tags": s.capability_tags,
                    "id": s.sample_id,
                }
                if s.metadata:
                    obj.update(s.metadata)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_lineage_from_jsonl(
    path: str,
    text_field: str = "text",
    generation_field: str = "generation",
    model_field: str = "source_model",
    capability_field: str = "capability_tags",
    id_field: str = "id",
) -> DatasetLineage:
    """从 JSONL 文件解析数据血统。

    每行 JSON 必须包含 text_field 和 generation_field。
    其他字段可选，缺失时使用默认值。
    """
    samples = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no + 1}: invalid JSON: {e}")

            text = obj.get(text_field, "")
            if not text:
                raise ValueError(f"Line {line_no + 1}: missing '{text_field}' field")

            samples.append(DataSample(
                text=text,
                generation=int(obj.get(generation_field, 0)),
                source_model=str(obj.get(model_field, "unknown")),
                capability_tags=list(obj.get(capability_field, [])),
                sample_id=str(obj.get(id_field, f"L{line_no + 1:05d}")),
                metadata={k: v for k, v in obj.items()
                          if k not in {text_field, generation_field, model_field,
                                       capability_field, id_field}},
            ))

    if not samples:
        raise ValueError(f"No samples found in {path}")

    return DatasetLineage(samples=samples)


def generate_synthetic_lineage(
    human_samples: list[str],
    n_generations: int = 3,
    decay_pattern: Optional[dict[str, float]] = None,
    executor_pattern: Optional[dict[str, dict[str, float]]] = None,
    capability_tags: Optional[list[str]] = None,
    u_shape_ei: bool = True,
) -> DatasetLineage:
    """生成模拟的多代合成数据血统（用于测试）。

    decay_pattern: {'math_reasoning': 0.40, ...}
        每个 capability 的 β 值。缺省使用差异化的默认 β。
    executor_pattern: {'math_reasoning': {'E-I': 0.7, 'E-II': 0.2, 'E-III': 0.1}, ...}
        每个 capability 的执行者构成。缺省从 capability tags 自动推断。
        也支持 '*' 通配符: {'*': {'E-I': 0.6, 'E-II': 0.3, 'E-III': 0.1}}
    capability_tags: 如提供，所有样本统一使用此标签列表（而非 auto_tag）
    """
    import random
    random.seed(42)

    # 异构默认衰减
    default_decay = {
        "math_reasoning": 0.40,
        "code_generation": 0.30,
        "logical_consistency": 0.35,
        "factual_knowledge": 0.20,
        "style_diversity": 0.15,
        "creative_writing": 0.12,
        "safety_alignment": 0.08,
        "general": 0.25,
    }
    betas = decay_pattern or default_decay

    # executor_pattern 缺省为 None（由 _apply_decay 从 tags 自动推断）
    exec_mix_map = executor_pattern or {}

    samples = []
    gen0_tags = []
    for i, text in enumerate(human_samples):
        tags = capability_tags if capability_tags else _auto_tag(text)
        gen0_tags.append(tags)
        samples.append(DataSample(
            text=text, generation=0, source_model="human",
            capability_tags=tags, sample_id=f"G0_{i:04d}",
        ))

    prev_gen = human_samples[:]
    prev_tags = list(gen0_tags)
    for gen in range(1, n_generations + 1):
        curr_gen = []
        for i, text in enumerate(prev_gen):
            tags = prev_tags[i]
            wildcard_default = betas.get("*", 0.25)
            sample_beta = max((betas.get(t, wildcard_default) for t in tags), default=wildcard_default)

            # 获取该样本的执行者构成
            wildcard_exec = exec_mix_map.get("*")
            exec_mix = None
            for t in tags:
                if t in exec_mix_map:
                    exec_mix = exec_mix_map[t]
                    break
            if exec_mix is None and wildcard_exec is not None:
                exec_mix = wildcard_exec

            degraded = _apply_decay(text, gen, sample_beta, tags, executor_mix=exec_mix, u_shape_ei=u_shape_ei)
            curr_gen.append(degraded)
        for i, text in enumerate(curr_gen):
            samples.append(DataSample(
                text=text, generation=gen,
                source_model=f"gen_model_v{gen}",
                capability_tags=prev_tags[i],
                sample_id=f"G{gen}_{i:04d}",
            ))
        prev_gen = curr_gen

    return DatasetLineage(samples=samples)


def _auto_tag(text: str) -> list[str]:
    """简单关键词匹配自动标注 capability tags。"""
    text_lower = text.lower()
    tags = []
    if any(w in text_lower for w in ["math", "calculate", "equation", "solve", "证明"]):
        tags.append("math_reasoning")
    if any(w in text_lower for w in ["code", "function", "python", "javascript", "编程"]):
        tags.append("code_generation")
    if any(w in text_lower for w in ["fact", "history", "science", "capital", "事实"]):
        tags.append("factual_knowledge")
    if any(w in text_lower for w in ["therefore", "because", "however", "逻辑"]):
        tags.append("logical_consistency")
    if any(w in text_lower for w in ["story", "poem", "creative", "小说"]):
        tags.append("creative_writing")
    if not tags:
        tags.append("general")
    return tags


# 按执行者类型分组的能力维度 → 决定衰减模式
_EI_CAPABILITIES = {"math_reasoning", "code_generation", "logical_consistency"}
_EII_CAPABILITIES = {"style_diversity", "creative_writing", "instruction_following",
                     "translation", "summarization", "general"}
_EIII_CAPABILITIES = {"factual_knowledge", "safety_alignment"}


def _apply_decay(
    text: str,
    generation: int,
    beta: float,
    tags: list[str],
    executor_mix: Optional[dict[str, float]] = None,
    u_shape_ei: bool = True,
) -> str:
    """模拟约束衰减：按执行者构成混合施加三种退化模式。

    executor_mix: {'E-I': 0.6, 'E-II': 0.3, 'E-III': 0.1}
    缺省时从 capability tags 自动推断。

    E-I（公理级）退化：逻辑连接词丢失 + 句子结构断裂
      - u_shape_ei=True: Gen1 结构增强 → Gen2 平台 → Gen3+ 退化（匹配真实递归生成）
    E-II（标度级）退化：风格均质化 + 重复 + 词汇匮乏
    E-III（边界级）退化：事实错误 + 数字扰动 + 专名侵蚀
    """
    import random
    words = text.split()
    if len(words) < 5:
        return text

    intensity = generation * beta

    # 确定执行者混合比例
    if executor_mix is not None:
        p_ei = executor_mix.get("E-I", 0.33)
        p_eii = executor_mix.get("E-II", 0.33)
        p_eiii = executor_mix.get("E-III", 0.34)
    else:
        ei_score = sum(1 for t in tags if t in _EI_CAPABILITIES)
        eii_score = sum(1 for t in tags if t in _EII_CAPABILITIES)
        eiii_score = sum(1 for t in tags if t in _EIII_CAPABILITIES)
        total = ei_score + eii_score + eiii_score
        if total == 0:
            p_ei, p_eii, p_eiii = 0.33, 0.33, 0.34
        else:
            p_ei = ei_score / total
            p_eii = eii_score / total
            p_eiii = eiii_score / total

    result = words[:]

    # ================================================================
    # E-I 退化：逻辑连接词丢失 + 句子结构断裂
    # U-shaped 模式：Gen1 结构增强 → Gen2 平台 → Gen3+ 退化
    # 匹配真实 Qwen2.5-7B 递归生成中观察到的行为
    # ================================================================
    ei_intensity = intensity * p_ei * 1.5

    logic_words = {"therefore", "because", "thus", "however", "hence", "consequently",
                   "moreover", "furthermore", "accordingly", "since", "then", "so",
                   "if", "but", "and", "or",
                   "因此", "所以", "然而", "因为", "从而", "于是", "故", "则"}
    logic_additions = ["therefore", "thus", "however", "furthermore", "consequently",
                       "moreover", "accordingly", "hence"]

    if u_shape_ei and generation <= 2:
        # Gen 1: 结构增强 — 插入逻辑连接词（模型规整化种子文本）
        if generation == 1 and ei_intensity > 0.05 and len(result) > 6:
            n_add = max(1, int(len(result) * ei_intensity * 0.1))
            for _ in range(n_add):
                insert_pos = random.randint(1, len(result) - 1)
                connector = random.choice(list(logic_additions))
                result.insert(insert_pos, connector)
        # Gen 2: 平台 — 轻度退化开始（少量连接词丢失）
        elif generation == 2 and ei_intensity > 0.2 and len(result) > 8:
            drop_prob = min(ei_intensity * 0.15, 0.3)
            n_drop = int(len(result) * ei_intensity * 0.04)
            indices_to_drop = set(random.sample(range(len(result)), min(n_drop, len(result) - 8)))
            result = [w for i, w in enumerate(result) if i not in indices_to_drop]
    else:
        # Gen 3+: 标准退化 — 逻辑连接词丢失 + 句子结构断裂
        if ei_intensity > 0.1:
            drop_prob = min(ei_intensity * 0.6, 0.85)
            result = [w for w in result
                      if w.lower() not in logic_words or random.random() > drop_prob]

    # 词序局部交换（推理链断裂）— 不删词，保持文本长度
    if ei_intensity > 0.3 and len(result) > 8:
        n_swaps = int(len(result) * ei_intensity * 0.12)
        for _ in range(max(1, n_swaps)):
            i, j = random.sample(range(len(result)), 2)
            if abs(i - j) <= 3:
                result[i], result[j] = result[j], result[i]

    # 轻量词丢失（Gen3+ 或非 U 形模式才执行）
    if not u_shape_ei or generation >= 3:
        if ei_intensity > 0.5:
            n_drop_ei = int(len(result) * ei_intensity * 0.08)
            n_drop_ei = max(0, min(n_drop_ei, max(0, len(result) - 8)))
            if n_drop_ei > 0 and len(result) > 8:
                indices = set(random.sample(range(len(result)), n_drop_ei))
                result = [w for i, w in enumerate(result) if i not in indices]

    # ================================================================
    # E-II 退化：风格均质化 + 重复 + 词汇截断
    # ================================================================
    eii_intensity = intensity * p_eii * 1.5

    # 短语重复（多次小规模重复，累积产生 bigram 重复信号）
    if eii_intensity > 0.08 and len(result) > 4:
        n_repeats = max(1, int(eii_intensity * 3))
        for _ in range(n_repeats):
            if len(result) < 4:
                break
            phrase_len = min(3, max(2, len(result) // 3))
            start = random.randint(0, len(result) - phrase_len)
            insert_pos = random.randint(0, len(result) - 1)
            result[insert_pos:insert_pos] = result[start:start + phrase_len]

    # 词汇多样性下降：用高频短词替代长词
    filler_words = ["the", "and", "is", "a", "to", "of", "it", "in", "this",
                    "that", "for", "with", "on", "as", "at", "by"]
    if eii_intensity > 0.15:
        for i in range(len(result)):
            if len(result[i]) > 5 and random.random() < eii_intensity * 0.12:
                result[i] = random.choice(filler_words)

    # 长词截断
    if eii_intensity > 0.2:
        for i in range(len(result)):
            if len(result[i]) > 7 and random.random() < eii_intensity * 0.15:
                result[i] = result[i][:max(3, int(len(result[i]) * 0.5))]

    # ================================================================
    # E-III 退化：事实侵蚀 + 数字扰动 + 专名退化
    # ================================================================
    eiii_intensity = intensity * p_eiii * 1.5

    # 专有名词小写化
    if eiii_intensity > 0.1:
        for i in range(len(result)):
            if len(result[i]) > 1 and result[i][0].isupper():
                if random.random() < eiii_intensity * 0.35:
                    result[i] = result[i].lower()

    # 数字扰动（四舍五入到整十/整百 — 精度丢失）
    if eiii_intensity > 0.08:
        for i in range(len(result)):
            w = result[i].strip('.,;:()[]{}')
            if w.isdigit() and random.random() < eiii_intensity * 0.3:
                try:
                    val = int(w)
                    # 精度退化：大数→整十，更大的→整百
                    if val >= 1000:
                        result[i] = str(round(val, -2))
                    elif val >= 100:
                        result[i] = str(round(val, -1))
                    else:
                        jitter = max(1, int(val * 0.1))
                        result[i] = str(val + random.randint(-jitter, jitter))
                except ValueError:
                    pass

    # 句首大写丢失（边界标记丢失，也是 E-III 指纹）
    if eiii_intensity > 0.25 and len(result) > 1:
        if result[0][0].isupper() and random.random() < 0.5:
            result[0] = result[0][0].lower() + result[0][1:]

    if not result:
        return text
    return " ".join(result)
