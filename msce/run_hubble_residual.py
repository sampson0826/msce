"""MSCE Hubble Tension — Sam's Residual Direction Diagnosis + Targeted Combination Search.
Step 1: Compute 8D residual vector from 6 single-proposal results
Step 2: Diagnose physical epoch from residual direction
Step 3: Select 3-4 targeted combinations
Step 4: Run MSCE on combinations
"""
import json, os, time, sys
sys.path.insert(0, os.path.dirname(__file__))
from product_engine import run_msce, run_single_model

HUBBLE_CONTEXT = """## 哈勃张力（Hubble Tension）核心数据

**两个主要测量：**
- Planck 2018 (CMB): H₀ = 67.4 ± 0.5 km/s/Mpc
- SH0ES 2022 (距离阶梯): H₀ = 73.0 ± 1.0 km/s/Mpc
- 差异：5.6 km/s/Mpc，约5σ显著性

**必须同时满足的观测约束：**
1. CMB温度各向异性功率谱（TT/TE/EE）——精确到0.1%
2. 重子声学振荡（BAO）尺度——从z=0.1到z=2.4
3. Ia型超新星距离模数（Pantheon+样本，1700+超新星）
4. 原初元素丰度（BBN：氘、氦-4丰度）
5. 大尺度结构S₈=σ₈√(Ω_m/0.3)（KiDS-1000/DES）
6. 宇宙年龄≥13.0 Gyr
7. 引力检验（太阳系约束）
8. 跨约束自洽性"""

PROPOSALS = {
    "A_early_dark_energy": {
        "name": "早期暗能量 (EDE)", "z_range": "early (z~3000→500)",
        "mechanism": "在z~3000处注入~5%额外暗能量，持续到z~500左右衰减",
        "parameters": "f_EDE ~ 0.05, z_c ~ 3000, n ~ 3",
        "claimed_effects": "H₀↑至~71-73",
        "known_challenges": "S₈张力可能加剧"
    },
    "B_modified_gravity": {
        "name": "修改引力 (MG)", "z_range": "late (z<1, solar system)",
        "mechanism": "在晚期宇宙修改广义相对论，局部有效引力常数变化",
        "parameters": "f_R0 ~ 10^-6 到 10^-4",
        "claimed_effects": "只在z<0.01内影响H₀测量",
        "known_challenges": "需满足太阳系检验"
    },
    "C_extra_neutrinos": {
        "name": "额外中微子 (ΔN_eff)", "z_range": "early (BBN, pre-recombination)",
        "mechanism": "引入额外辐射自由度ΔN_eff≈0.3-0.5",
        "parameters": "ΔN_eff ~ 0.3-0.5",
        "claimed_effects": "H₀↑至~70-71",
        "known_challenges": "BBN氦-4丰度约束"
    },
    "D_decaying_dark_matter": {
        "name": "衰变暗物质 (DDM)", "z_range": "early→intermediate (z~10^4→10^2)",
        "mechanism": "部分暗物质衰变为暗辐射",
        "parameters": "Γ ~ (10-100 Gyr)^-1",
        "claimed_effects": "H₀↑至~70-72",
        "known_challenges": "再电离历史、CMB透镜谱"
    },
    "E_local_void": {
        "name": "局部空洞", "z_range": "very late (z~0.05, local 200Mpc)",
        "mechanism": "银河系位于低密度区，局部膨胀率偏高",
        "parameters": "δ ~ -0.3, 半径 ~ 200-300 Mpc",
        "claimed_effects": "不需要修改ΛCDM物理",
        "known_challenges": "振幅不足以解释全部张力"
    },
    "F_systematic_error": {
        "name": "系统误差", "z_range": "N/A (observational)",
        "mechanism": "Planck或SH0ES存在未识别系统误差",
        "parameters": "无自由参数",
        "claimed_effects": "如果一方有误差，张力消失",
        "known_challenges": "JWST已验证造父变星"
    }
}

CONSTRAINT_NAMES = [
    "cmb_power_spectrum",
    "bao_scale",
    "supernova_hubble_diagram",
    "bbn_abundances",
    "s8_tension",
    "universe_age",
    "gravity_tests",
    "cross_constraint_consistency"
]

# Physical epoch mapping for each constraint
CONSTRAINT_EPOCH = {
    "cmb_power_spectrum": "early (z~1100, recombination)",
    "bao_scale": "early (z~1100, drag epoch)",
    "supernova_hubble_diagram": "late (z<2, cosmic expansion)",
    "bbn_abundances": "very early (z~10^9, nucleosynthesis)",
    "s8_tension": "late (z<2, large-scale structure)",
    "universe_age": "global (integrated expansion)",
    "gravity_tests": "solar system (z=0)",
    "cross_constraint_consistency": "global (all epochs)"
}

RESIDUAL_CHECK_PROMPT = """你是一个精确的宇宙学约束验证器。对给定的H₀张力解决方案，逐项检查它是否与已知观测约束一致。

## 8项约束：
1. **CMB温度功率谱**：是否保持声学峰位置（θ_s）、峰高比？（Planck TT谱精确到~1%）
2. **BAO尺度**：是否保持共动声视界r_d在147-148 Mpc？（DESI 2024: r_d = 147.5±0.3 Mpc）
3. **超新星距离模数**：是否与Pantheon+的Hubble图一致？（μ(z)在z<2范围内）
4. **BBN原初核合成**：是否保持氘丰度D/H≈2.55×10⁻⁵和氦-4质量分数Y_p≈0.245？
5. **大尺度结构S₈**：方案是否使S₈偏离KiDS-1000/DES观测值S₈≈0.77±0.02？
6. **宇宙年龄**：方案预测的宇宙年龄是否≥13.0 Gyr？
7. **引力检验**：是否满足太阳系检验？
8. **跨约束自洽性**：参数调节是否同时破坏其他观测量的预测？

## 判断标准：
- pass: 约束满足
- violation: 明确矛盾（给出具体数值和观测限制）
- tension: 在1.5-3σ范围内存在张力

## 输出格式（严格JSON，只输出JSON）：
{
  "cmb_power_spectrum": {"status": "pass", "detail": "..."},
  "bao_scale": {"status": "violation", "detail": "..."},
  "supernova_hubble_diagram": {"status": "pass", "detail": "..."},
  "bbn_abundances": {"status": "tension", "detail": "..."},
  "s8_tension": {"status": "pass", "detail": "..."},
  "universe_age": {"status": "pass", "detail": "..."},
  "gravity_tests": {"status": "pass", "detail": "..."},
  "cross_constraint_consistency": {"status": "tension", "detail": "..."},
  "overall_judgment": "...",
  "pass_count": 0,
  "violation_count": 0
}"""


def status_to_score(status):
    """Convert constraint status to numerical score."""
    return {"pass": 0, "tension": 1, "violation": 2, "unknown": 1}.get(status, 1)


def run_residual_diagnosis():
    """Step 1: Re-extract clean constraint statuses, compute residual vector."""
    print("=" * 70)
    print("STEP 1: Residual Direction Diagnosis")
    print("=" * 70)

    all_statuses = {}
    for key, info in PROPOSALS.items():
        question = f"""{HUBBLE_CONTEXT}

## 待验证方案：{info['name']}

**机制：** {info['mechanism']}
**关键参数：** {info['parameters']}
**声称的效果：** {info['claimed_effects']}
**已知挑战：** {info['known_challenges']}

请对此方案进行8项约束一致性检查。"""

        print(f"\nChecking: {info['name']}...")
        try:
            raw = run_single_model(RESIDUAL_CHECK_PROMPT + "\n\n" + question, "mkeai", "gpt-5.5")
            # Extract JSON from response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw[json_start:json_end])
                all_statuses[key] = parsed
                print(f"  pass={parsed.get('pass_count',0)}, violation={parsed.get('violation_count',0)}")
            else:
                print(f"  [WARN] Could not extract JSON from response")
                all_statuses[key] = {"error": "no_json", "raw": raw[:200]}
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_statuses[key] = {"error": str(e)}

    # Compute residual vector
    print("\n" + "-" * 60)
    print("8D Residual Vector (avg deviation from 'all pass'):")
    print("-" * 60)

    residual = {}
    per_proposal_scores = {}
    for cname in CONSTRAINT_NAMES:
        scores = []
        for key in PROPOSALS:
            r = all_statuses.get(key, {})
            if "error" not in r:
                status = r.get(cname, {}).get("status", "unknown")
                scores.append(status_to_score(status))
        if scores:
            avg = sum(scores) / len(scores)
            residual[cname] = {"avg_score": round(avg, 3), "n": len(scores), "epoch": CONSTRAINT_EPOCH[cname]}
            per_proposal_scores[cname] = scores

    # Display residual vector sorted by score
    for cname, info in sorted(residual.items(), key=lambda x: -x[1]["avg_score"]):
        bar = "█" * int(info["avg_score"] * 10) + "░" * (20 - int(info["avg_score"] * 10))
        print(f"  {cname:<32} [{bar}] {info['avg_score']:.2f}  ({info['epoch']})")

    # Epoch analysis
    early_constraints = ["cmb_power_spectrum", "bao_scale", "bbn_abundances"]
    late_constraints = ["supernova_hubble_diagram", "s8_tension", "universe_age", "gravity_tests"]
    global_constraints = ["cross_constraint_consistency"]

    early_residual = sum(residual[c]["avg_score"] for c in early_constraints) / len(early_constraints)
    late_residual = sum(residual[c]["avg_score"] for c in late_constraints) / len(late_constraints)
    global_residual = sum(residual[c]["avg_score"] for c in global_constraints) / len(global_constraints)

    print(f"\nEpoch Analysis:")
    print(f"  Early universe (z>1000) residual:  {early_residual:.2f}")
    print(f"  Late universe (z<2) residual:     {late_residual:.2f}")
    print(f"  Global consistency residual:      {global_residual:.2f}")

    return all_statuses, residual, early_residual, late_residual, global_residual


def design_combinations(residual, early_residual, late_residual, global_residual):
    """Step 2: Based on residual direction, select targeted combinations."""
    print("\n" + "=" * 70)
    print("STEP 2: Combination Design from Residual Direction")
    print("=" * 70)

    # Find which constraints have highest residual
    high_residual_constraints = [c for c, info in residual.items() if info["avg_score"] >= 1.0]

    print(f"\nHigh-residual constraints (avg score >= 1.0):")
    for c in high_residual_constraints:
        print(f"  - {c} ({CONSTRAINT_EPOCH[c]}): {residual[c]['avg_score']:.2f}")

    # Analysis: which proposals fail on which constraints (complementarity check)
    print(f"\nComplementarity analysis:")
    print(f"  Early residual: {early_residual:.2f} — need early-universe mechanism")
    print(f"  Late residual:  {late_residual:.2f} — need late-universe mechanism")

    # Sam's recommended combinations based on epoch diagnosis
    combinations = []

    if early_residual > 0.5 and late_residual > 0.5:
        # Both epochs need coverage → cross-epoch combinations
        print(f"\n  → Both epochs show significant residual → cross-epoch combinations")
        combinations = [
            {
                "id": "combo_1",
                "name": "EDE + 局部空洞 (early+late cross-epoch)",
                "mechanism": "EDE在z~3000注入额外能量减小声视界→H₀↑至~71，局部空洞（δ~-0.3, 200Mpc）提供额外~2km/s/Mpc→总H₀≈73",
                "proposals": ["A_early_dark_energy", "E_local_void"],
                "rationale": "EDE解决大部分H₀偏移但破坏BAO/S₈，局部空洞不影响BAO且提供额外局部膨胀→物理机制正交，互补性最高"
            },
            {
                "id": "combo_2",
                "name": "ΔN_eff + 局部空洞 (early radiation + late void)",
                "mechanism": "ΔN_eff≈0.2-0.3增加早期辐射密度+局部空洞提供局部膨胀偏置",
                "proposals": ["C_extra_neutrinos", "E_local_void"],
                "rationale": "降低ΔN_eff到BBN可容忍范围（<0.3），不足部分由局部空洞补充→避免各自单独使用时的主要矛盾"
            },
            {
                "id": "combo_3",
                "name": "EDE + ΔN_eff (dual early mechanism, fine-tuned)",
                "mechanism": "EDE（f_EDE~0.03）+ΔN_eff≈0.2，两者剂量减半→各自副作用降低但协同效果保持",
                "proposals": ["A_early_dark_energy", "C_extra_neutrinos"],
                "rationale": "EDE的主要问题在S₈，ΔN_eff的问题在BBN，降低各自剂量→边际效应可能落在约束容忍范围内"
            },
            {
                "id": "combo_4",
                "name": "DDM + 局部空洞 (decaying DM + void)",
                "mechanism": "DDM（Γ~50Gyr⁻¹）+局部空洞，DDM提供平滑早期效应+空洞提供晚期局部偏置",
                "proposals": ["D_decaying_dark_matter", "E_local_void"],
                "rationale": "DDM对CMB的影响比EDE更平滑（无尖锐注入），可能与局部空洞正交互补"
            },
        ]
    elif early_residual > 0.6:
        print(f"\n  → Early universe dominates → dual early mechanism combinations")
        combinations = [
            {
                "id": "combo_1",
                "name": "EDE + ΔN_eff (dual early, reduced dose)",
                "mechanism": "f_EDE~0.03 + ΔN_eff≈0.2，各取半剂量",
                "proposals": ["A_early_dark_energy", "C_extra_neutrinos"],
                "rationale": "两个早期机制各降剂量→副作用可能落在约束容忍范围内"
            },
            {
                "id": "combo_2",
                "name": "DDM + ΔN_eff (smooth early + radiation)",
                "mechanism": "DDM提供平滑早期能量注入+ΔN_eff精细调节",
                "proposals": ["D_decaying_dark_matter", "C_extra_neutrinos"],
                "rationale": "DDM比EDE更平滑，配合低剂量ΔN_eff可能避开BBN约束"
            },
            {
                "id": "combo_3",
                "name": "EDE + DDM (two-stage early injection)",
                "mechanism": "EDE在z~3000注入+DDM在z~10^4→10^2持续衰变",
                "proposals": ["A_early_dark_energy", "D_decaying_dark_matter"],
                "rationale": "两阶段能量注入可能提供更平滑的H(z)修正"
            },
        ]
    else:
        print(f"\n  → Late universe or global → mixed approach")
        combinations = [
            {
                "id": "combo_1",
                "name": "EDE + MG (early + local gravity)",
                "mechanism": "EDE提供全局H₀偏移+MG提供局部效应微调",
                "proposals": ["A_early_dark_energy", "B_modified_gravity"],
                "rationale": "互补性最高：EDE改早期，MG改晚期局部"
            },
            {
                "id": "combo_2",
                "name": "MG + 局部空洞 (dual late mechanism)",
                "mechanism": "修改引力+f(R)+局部空洞分别提供不同尺度的晚期效应",
                "proposals": ["B_modified_gravity", "E_local_void"],
                "rationale": "两个晚期机制在不同尺度上作用"
            },
            {
                "id": "combo_3",
                "name": "ΔN_eff + MG (early radiation + late gravity)",
                "mechanism": "低剂量ΔN_eff+局部引力修正",
                "proposals": ["C_extra_neutrinos", "B_modified_gravity"],
                "rationale": "早期+晚期跨时代互补"
            },
        ]

    print(f"\nSelected {len(combinations)} combinations for MSCE verification:")
    for c in combinations:
        print(f"  {c['id']}: {c['name']}")

    return combinations


def run_combination_checks(combinations, all_statuses, residual):
    """Step 3: Run MSCE on selected combinations."""
    print("\n" + "=" * 70)
    print("STEP 3: MSCE Combination Verification")
    print("=" * 70)

    # Build residual summary for context
    residual_summary = "\n".join([
        f"  {cname}: avg_deviation={info['avg_score']:.2f} (epoch: {info['epoch']})"
        for cname, info in sorted(residual.items(), key=lambda x: -x[1]["avg_score"])
    ])

    results = {}
    for combo in combinations:
        p1_key, p2_key = combo["proposals"]
        p1 = PROPOSALS[p1_key]
        p2 = PROPOSALS[p2_key]

        question = f"""{HUBBLE_CONTEXT}

## 残差方向诊断结果
6个单方案在8项约束上的平均残差向量：
{residual_summary}

诊断：单因子空间无解。需跨时代组合。

## 待验证组合方案：{combo['name']}

**组合机制：** {combo['mechanism']}

**组成方案1：{p1['name']}**
- 机制：{p1['mechanism']}
- 参数：{p1['parameters']}
- 作用红移范围：{p1['z_range']}

**组成方案2：{p2['name']}**
- 机制：{p2['mechanism']}
- 参数：{p2['parameters']}
- 作用红移范围：{p2['z_range']}

**选择理由：** {combo['rationale']}

请用MSCE的约束传播方法验证：这个组合能否在保持所有8项观测约束一致的前提下，将H₀提升至~73 km/s/Mpc？

请特别注意：
1. 两个机制的物理参数是否可独立调节？
2. 组合后是否会产生新的约束冲突（不在单方案中出现的）？
3. 组合方案是否比任何单一方案表现更好？"""

        print(f"\nAnalyzing: {combo['name']}...")
        try:
            result = run_msce(question, domain="constraint_propagation")
            results[combo["id"]] = {
                "combo_name": combo["name"],
                "msce_confidence": result.get("confidence", 0),
                "msce_disagreement": result.get("disagreement", 0),
                "msce_top_answer": result.get("top_answer", "")[:500],
                "msce_elapsed": result.get("elapsed_time", 0),
                "proposals": combo["proposals"],
            }
            print(f"  MSCE conf={result.get('confidence', 0):.3f}, disag={result.get('disagreement', 0):.3f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[combo["id"]] = {"error": str(e), "combo_name": combo["name"]}

    return results


def main():
    t_start = time.time()

    # Step 1: Residual diagnosis
    all_statuses, residual, early_res, late_res, global_res = run_residual_diagnosis()

    # Step 2: Design combinations
    combinations = design_combinations(residual, early_res, late_res, global_res)

    # Step 3: Run combination checks
    combo_results = run_combination_checks(combinations, all_statuses, residual)

    # Summary
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("HUBBLE RESIDUAL DIAGNOSIS + COMBINATION SEARCH — SUMMARY")
    print("=" * 70)
    print(f"\nResidual Direction:")
    print(f"  Early universe (z>1000): {early_res:.2f}")
    print(f"  Late universe (z<2):    {late_res:.2f}")
    print(f"  Global consistency:     {global_res:.2f}")

    print(f"\nCombination Results:")
    for cid, cr in combo_results.items():
        print(f"  {cr.get('combo_name', cid)}: conf={cr.get('msce_confidence', 0):.3f}")

    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    output = {
        "timestamp": time.time(),
        "analysis_type": "hubble_residual_diagnosis_with_combinations",
        "method": "Sam's residual direction → targeted combination search",
        "residual_diagnosis": {
            "per_constraint": residual,
            "epoch_summary": {
                "early_universe": round(early_res, 3),
                "late_universe": round(late_res, 3),
                "global_consistency": round(global_res, 3),
            },
            "interpretation": "High early+late → need cross-epoch combinations" if (early_res > 0.5 and late_res > 0.5) else (
                "Early-dominant → dual early mechanisms" if early_res > 0.6 else "Late/global → mixed approach"
            )
        },
        "combinations_checked": [
            {
                "id": cid,
                "name": cr.get("combo_name", ""),
                "proposals": cr.get("proposals", []),
                "msce_confidence": cr.get("msce_confidence", 0),
                "msce_disagreement": cr.get("msce_disagreement", 0),
                "msce_top_answer": cr.get("msce_top_answer", "")[:500],
                "msce_elapsed": cr.get("msce_elapsed", 0),
            }
            for cid, cr in combo_results.items()
        ],
        "single_proposal_statuses": {
            k: {"pass_count": v.get("pass_count", 0), "violation_count": v.get("violation_count", 0),
                "overall_judgment": v.get("overall_judgment", "")[:300]}
            for k, v in all_statuses.items() if "error" not in v
        },
        "summary": {
            "total_time_s": round(total_time, 1),
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/hubble_residual_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
