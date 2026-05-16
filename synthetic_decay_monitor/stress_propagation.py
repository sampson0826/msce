"""
应力传播模拟器 — 能力维度拓扑图 + 级联崩溃预测。

切割一条约束边 → 应力沿拓扑传播 → 最脆弱邻接边先断裂。
断裂点不在冲击源，在拓扑最脆弱处。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CapabilityEdge:
    source: str
    target: str
    coupling_strength: float = 0.5    # 0-1，耦合强度
    stress_capacity: float = 1.0      # 最大承压
    current_stress: float = 0.0       # 当前应力
    is_broken: bool = False


@dataclass
class CascadeEvent:
    generation: int
    broken_capability: str
    stress_source: str          # 哪个能力维度的断裂触发了这个
    accumulated_stress: float
    remaining_capacity: float
    downstream_affected: list[str] = field(default_factory=list)


# 预定义能力维度依赖图
# 格式：{source: [(target, coupling_strength, stress_capacity), ...]}
DEFAULT_CAPABILITY_GRAPH: dict[str, list[tuple[str, float, float]]] = {
    "math_reasoning": [
        ("code_generation", 0.85, 0.60),
        ("logical_consistency", 0.90, 0.50),
        ("factual_knowledge", 0.30, 0.80),
    ],
    "code_generation": [
        ("logical_consistency", 0.70, 0.60),
        ("instruction_following", 0.60, 0.70),
    ],
    "logical_consistency": [
        ("factual_knowledge", 0.50, 0.70),
        ("summarization", 0.40, 0.80),
        ("translation", 0.20, 0.90),
    ],
    "factual_knowledge": [
        ("summarization", 0.40, 0.80),
        ("creative_writing", 0.15, 0.90),
    ],
    "style_diversity": [
        ("creative_writing", 0.60, 0.60),
        ("translation", 0.50, 0.70),
        ("summarization", 0.40, 0.80),
    ],
    "safety_alignment": [
        ("instruction_following", 0.70, 0.50),
        ("creative_writing", 0.30, 0.70),
    ],
    "instruction_following": [
        ("code_generation", 0.50, 0.70),
        ("summarization", 0.45, 0.80),
    ],
    "creative_writing": [
        ("summarization", 0.30, 0.80),
    ],
    "translation": [],
    "summarization": [],
}


class CapabilityTopology:
    def __init__(self, graph: Optional[dict] = None):
        self.graph = graph or DEFAULT_CAPABILITY_GRAPH
        self._edges = self._build_edges()
        self._node_stress: dict[str, float] = {}
        self._broken: set[str] = set()

    def _build_edges(self) -> dict[tuple[str, str], CapabilityEdge]:
        edges = {}
        for src, targets in self.graph.items():
            for tgt, coupling, capacity in targets:
                key = (src, tgt)
                edges[key] = CapabilityEdge(
                    source=src,
                    target=tgt,
                    coupling_strength=coupling,
                    stress_capacity=capacity,
                )
        return edges

    @property
    def all_capabilities(self) -> list[str]:
        nodes = set(self.graph.keys())
        for targets in self.graph.values():
            for tgt, _, _ in targets:
                nodes.add(tgt)
        return sorted(nodes)

    def simulate_cascade(
        self,
        initial_break: str,
        stability_map: dict[str, float],
        max_steps: int = 10,
    ) -> list[CascadeEvent]:
        """模拟级联崩溃。

        Args:
            initial_break: 最先断裂的能力维度
            stability_map: {capability: S_n} 当前各能力维度的稳定性
            max_steps: 最大模拟步数

        Returns:
            级联事件序列
        """
        self._node_stress = {cap: 0.0 for cap in self.all_capabilities}
        self._broken = set()
        events: list[CascadeEvent] = []

        # 初始断裂
        self._broken.add(initial_break)
        events.append(CascadeEvent(
            generation=0,
            broken_capability=initial_break,
            stress_source="initial",
            accumulated_stress=0.0,
            remaining_capacity=0.0,
            downstream_affected=self._get_downstream(initial_break),
        ))

        # 传播初始应力到下游
        self._propagate_stress(initial_break, 1.0)

        step = 1
        while step <= max_steps:
            # 找到应力最大且未断裂的节点
            candidates = {
                cap: stress
                for cap, stress in self._node_stress.items()
                if cap not in self._broken and stress > 0
            }

            if not candidates:
                break

            # 最脆弱 = 应力/稳定性 最高
            fragility = {
                cap: stress / max(stability_map.get(cap, 0.3), 0.01)
                for cap, stress in candidates.items()
            }

            next_break = max(fragility, key=fragility.get)

            # 检查是否真的断裂（超过承压）
            incoming_edges = [
                e for (s, t), e in self._edges.items()
                if t == next_break and s in self._broken
            ]
            total_stress = sum(e.coupling_strength * e.current_stress for e in incoming_edges)

            if len(incoming_edges) > 0:
                avg_capacity = np.mean([e.stress_capacity for e in incoming_edges])
            else:
                avg_capacity = 1.0

            if total_stress > avg_capacity * 0.6:
                self._broken.add(next_break)
                events.append(CascadeEvent(
                    generation=step,
                    broken_capability=next_break,
                    stress_source=",".join(
                        [e.source for e in self._edges.values()
                         if e.target == next_break]
                    ),
                    accumulated_stress=total_stress,
                    remaining_capacity=avg_capacity - total_stress,
                    downstream_affected=self._get_downstream(next_break),
                ))
                self._propagate_stress(next_break, 0.7)

            step += 1

        return events

    def find_weakest_edge(self, stability_map: dict[str, float]) -> dict:
        """定位拓扑中最脆弱的边。

        脆弱度 = 耦合强度 * (1 - 目标稳定性) / 承压
        """
        fragility_scores = {}
        for (src, tgt), edge in self._edges.items():
            tgt_stability = stability_map.get(tgt, 0.5)
            fragility = (
                edge.coupling_strength * (1 - tgt_stability) / max(edge.stress_capacity, 0.01)
            )
            fragility_scores[f"{src} → {tgt}"] = {
                "edge": f"{src} → {tgt}",
                "fragility": fragility,
                "coupling_strength": edge.coupling_strength,
                "target_stability": tgt_stability,
                "stress_capacity": edge.stress_capacity,
            }

        ranked = sorted(fragility_scores.values(), key=lambda x: -x["fragility"])
        return ranked[0] if ranked else {}

    def _propagate_stress(self, source: str, multiplier: float = 1.0):
        """将源断裂的应力传播到下游邻接边。"""
        stress_amplitude = multiplier * 0.85  # 传播衰减系数

        for (src, tgt), edge in self._edges.items():
            if src == source and tgt not in self._broken:
                edge.current_stress += stress_amplitude * edge.coupling_strength
                self._node_stress[tgt] += stress_amplitude * edge.coupling_strength

    def _get_downstream(self, capability: str) -> list[str]:
        return [tgt for src, tgt in self._edges if src == capability]


def run_cascade_analysis(
    collapse_order: list[dict],
    stability_map: Optional[dict[str, float]] = None,
) -> dict:
    """端到端级联分析。

    Args:
        collapse_order: 从 DecayEngine.get_collapse_order() 获取的崩溃排序
        stability_map: {capability: S_n}，如未提供则从 collapse_order 推导
    """
    if not collapse_order:
        return {"error": "no collapse data"}

    if stability_map is None:
        stability_map = {
            c["capability"]: c["current_S_n"]
            for c in collapse_order
            if "current_S_n" in c
        }

    topology = CapabilityTopology()

    first_to_collapse = collapse_order[0]["capability"]

    cascade = topology.simulate_cascade(first_to_collapse, stability_map)

    weakest = topology.find_weakest_edge(stability_map)

    return {
        "first_to_collapse": first_to_collapse,
        "collapse_order": [c["capability"] for c in collapse_order[:5]],
        "cascade_events": [
            {
                "step": e.generation,
                "broken": e.broken_capability,
                "stress_source": e.stress_source,
                "accumulated_stress": e.accumulated_stress,
                "downstream_affected": e.downstream_affected,
            }
            for e in cascade
        ],
        "weakest_edge": weakest,
        "total_cascade_steps": len(cascade),
        "full_collapse_set": sorted(list(topology._broken)),
        "summary": _cascade_summary(cascade, weakest),
    }


def _cascade_summary(events: list[CascadeEvent], weakest: dict) -> str:
    if not events:
        return "No cascade predicted."
    parts = [
        f"Initial break: {events[0].broken_capability}.",
    ]
    if len(events) > 1:
        cascade_path = " → ".join([e.broken_capability for e in events])
        parts.append(f"Cascade path: {cascade_path}.")
    else:
        parts.append("No subsequent cascade — stress contained.")
    if weakest:
        parts.append(
            f"Weakest structural edge: {weakest['edge']} "
            f"(fragility={weakest['fragility']:.3f})."
        )
    return " ".join(parts)
