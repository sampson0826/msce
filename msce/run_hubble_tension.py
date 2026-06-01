"""MSCE Hubble Tension Analysis — multi-constraint consistency verification.
Tests whether MSCE can identify hidden contradictions in proposed H0 resolutions.
"""
import json, os, time, sys
sys.path.insert(0, os.path.dirname(__file__))
from product_engine import run_msce, run_single_model, get_client

# ═══════════════════════════════════════════════════════════════════════════
# Hubble Tension: H0 = 67.4±0.5 (Planck CMB) vs 73.0±1.0 (SH0ES distance ladder)
# Statistical significance: ~5σ — unlikely to be statistical fluke
# Key puzzle: any resolution must satisfy ALL of these constraints simultaneously
# ═══════════════════════════════════════════════════════════════════════════

HUBBLE_CONTEXT = """## 哈勃张力（Hubble Tension）核心数据

**两个主要测量：**
- Planck 2018 (CMB): H₀ = 67.4 ± 0.5 km/s/Mpc
- SH0ES 2022 (距离阶梯): H₀ = 73.0 ± 1.0 km/s/Mpc
- 差异：5.6 km/s/Mpc，约5σ显著性

**独立交叉验证（支持较高值）：**
- H0LiCOW (强引力透镜时间延迟): H₀ = 73.3 ± 1.8
- Megamaser (水脉泽几何测距): H₀ = 73.9 ± 3.0
- SBF (表面亮度涨落): H₀ = 73.3 ± 2.5
- TDCOSMO (引力透镜): H₀ = 74.5 ± 5.8

**支持较低值的测量：**
- DESI 2024 (BAO+CMB): H₀ = 68.5 ± 0.7
- Planck+lensing+BAO联合: H₀ = 67.6 ± 0.4

**必须同时满足的观测约束：**
1. CMB温度各向异性功率谱（TT/TE/EE）——精确到0.1%
2. 重子声学振荡（BAO）尺度——从z=0.1到z=2.4
3. Ia型超新星距离模数（Pantheon+样本，1700+超新星）
4. 大尺度结构（弱引力透镜，KiDS/DES）
5. 原初元素丰度（BBN：氘、氦-4丰度）
6. 宇宙年龄≥13.0 Gyr（球状星团年龄下限）"""

# ═══════════════════════════════════════════════════════════════════════════
# Six major resolution proposals — each tested for multi-constraint consistency
# ═══════════════════════════════════════════════════════════════════════════

PROPOSALS = {
    "A_early_dark_energy": {
        "name": "早期暗能量 (Early Dark Energy, EDE)",
        "mechanism": "在z~3000处注入~5%额外暗能量，持续到z~500左右衰减。这使CMB退耦时的宇宙膨胀率增大→声视界减小→推导出的H₀升高，从而与SH0ES一致。",
        "parameters": "f_EDE ~ 0.05, z_c ~ 3000, 衰减指数 n ~ 3",
        "claimed_effects": "H₀↑至~71-73，不破坏CMB功率谱主峰位置",
        "known_challenges": "S₈张力可能加剧（σ₈预测偏高），可能与DES弱引力透镜数据冲突"
    },
    "B_modified_gravity": {
        "name": "修改引力 (Modified Gravity / f(R) / 大质量引力)",
        "mechanism": "在晚期宇宙（z<1）修改广义相对论，使距离阶梯测量的局部H₀偏高但全局膨胀率不变。相当于局部有效引力常数变化→造父变星周期-光度关系偏移。",
        "parameters": "f_R0 ~ 10^-6 到 10^-4, 或引力子质量 m_g",
        "claimed_effects": "只在z<0.01内影响H₀测量，不影响CMB和BAO",
        "known_challenges": "需满足太阳系检验（Cassini, 月球激光测距），可能破坏大尺度结构增长速率"
    },
    "C_extra_neutrinos": {
        "name": "额外中微子/暗辐射 (ΔN_eff + m_sterile)",
        "mechanism": "引入额外辐射自由度ΔN_eff≈0.3-0.5（如惰性中微子），增加CMB退耦前的辐射能量密度→声视界减小→推导H₀升高。",
        "parameters": "ΔN_eff ~ 0.3-0.5, 可能 m_sterile ~ 0.1-1 eV",
        "claimed_effects": "H₀↑至~70-71，同时保持CMB功率谱形状",
        "known_challenges": "BBN氦-4丰度约束ΔN_eff<0.3（95%CL），可能与氘丰度冲突。CMB阻尼尾可能检测到额外辐射平滑效应。"
    },
    "D_decaying_dark_matter": {
        "name": "衰变暗物质 (Decaying Dark Matter, DDM)",
        "mechanism": "部分暗物质（~5-10%）在z~10^4到z~10^2之间衰变为暗辐射。衰变产物增加晚期辐射密度→减小声视界→H₀升高。",
        "parameters": "衰变比例Γ ~ (10-100 Gyr)^-1, 衰变红移z_decay",
        "claimed_effects": "H₀↑至~70-72，对CMB的影响与EDE类似但更平滑",
        "known_challenges": "衰变热可能破坏再电离历史；扰动演化可能与Planck CMB透镜谱冲突"
    },
    "E_local_void": {
        "name": "局部空洞假说 (Local Void / 局部膨胀率偏高)",
        "mechanism": "银河系位于一个~200Mpc半径的低密度区（局部空洞），空洞内的局部膨胀率高于宇宙平均。造父变星→Ia超新星的距离阶梯全在空洞内→测得的H₀是局部值不是全局值。",
        "parameters": "空洞密度对比δ ~ -0.3, 半径 ~ 200-300 Mpc",
        "claimed_effects": "不需要修改ΛCDM物理，纯观测效应→H₀表观张力",
        "known_challenges": "现有巡天（2M++）显示局部空洞不足以解释全部张力（最多贡献~1-2km/s/Mpc）。需要精细调谐空洞形状。"
    },
    "F_systematic_error": {
        "name": "未被发现的系统误差 (Unknown Systematics)",
        "mechanism": "Planck或SH0ES其中一方（或双方）存在未被识别的系统误差。可能来源：CMB前景扣除、造父变星周光关系的金属丰度依赖、SNe Ia标准化中的人口漂移。",
        "parameters": "无自由参数——需要更精确的独立测量来验证",
        "claimed_effects": "如果只有一方有系统误差，张力自然消失",
        "known_challenges": "JWST已独立验证造父变星周期-光度关系（2023年结果支持SH0ES），降低了系统误差假说的可能性"
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# Constraint consistency check — the core analysis
# ═══════════════════════════════════════════════════════════════════════════

CONSTRAINT_CHECK_PROMPT = """你是一个精确的宇宙学约束验证器。对给定的H₀张力解决方案，逐项检查它是否与已知观测约束一致。

## 需要检查的约束：
1. **CMB温度功率谱**：方案是否保持声学峰位置（θ_s）、峰高比（暗物质/重子比）不变？（Planck TT谱精确到~1%）
2. **BAO尺度**：是否保持共动声视界r_d在147-148 Mpc？（DESI 2024 BAO测量约束r_d = 147.5±0.3 Mpc）
3. **超新星距离模数**：是否与Pantheon+的Hubble图一致？（μ(z)在z<2范围内）
4. **BBN原初核合成**：是否保持氘丰度D/H≈2.55×10⁻⁵和氦-4质量分数Y_p≈0.245？（额外辐射会改变BBN预测）
5. **大尺度结构S₈=σ₈√(Ω_m/0.3)**：方案是否使S₈偏离KiDS-1000/DES观测值S₈≈0.77±0.02？（EDE类方案通常会增大S₈）
6. **宇宙年龄**：方案预测的宇宙年龄是否≥13.0 Gyr？
7. **引力检验**：如果方案修改引力，是否满足太阳系检验？（|f_R0| < 10⁻⁶约束）
8. **跨约束自洽性**：如果方案通过调节宇宙学参数实现H₀↑，该参数调节是否同时破坏其他观测量的预测？

## 判断标准：
- pass: 约束满足或在观测误差范围内
- violation: 明确的矛盾（给出具体数值和观测限制）
- tension: 在1.5-3σ范围内存在张力但非明确矛盾
- unknown: 需要进一步计算才能确定

## 输出格式（严格JSON）：
{
  "cmb_power_spectrum": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "bao_scale": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "supernova_hubble_diagram": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "bbn_abundances": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "s8_tension": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "universe_age": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "gravity_tests": {"status": "pass/violation/tension/unknown", "detail": "具体分析"},
  "cross_constraint_consistency": {"status": "pass/violation/tension/unknown", "detail": "跨约束自洽性分析"},
  "overall_judgment": "该方案的核心矛盾是什么？（如有）",
  "pass_count": 0,
  "violation_count": 0
}

只输出JSON。"""


def analyze_proposal(proposal_key, proposal_info):
    """Run MSCE on a single proposal: multi-constraint consistency check."""
    question = f"""{HUBBLE_CONTEXT}

## 待验证方案：{proposal_info['name']}

**机制：** {proposal_info['mechanism']}

**关键参数：** {proposal_info['parameters']}

**声称的效果：** {proposal_info['claimed_effects']}

**已知挑战：** {proposal_info['known_challenges']}

请对此方案进行8项约束一致性检查。"""

    print(f"\n{'='*70}")
    print(f"Analyzing: {proposal_info['name']}")
    print(f"{'='*70}")

    # Run MSCE with constraint_propagation domain (most relevant)
    result = run_msce(question, domain="constraint_propagation")

    # Also get a focused single-model analysis
    try:
        constraint_check = run_single_model(
            CONSTRAINT_CHECK_PROMPT + "\n\n" + question,
            "mkeai", "gpt-5.5"
        )
    except Exception as e:
        constraint_check = f"[ERROR: {e}]"

    return {
        "proposal_key": proposal_key,
        "proposal_name": proposal_info["name"],
        "msce_confidence": result.get("confidence", 0),
        "msce_disagreement": result.get("disagreement", 0),
        "msce_top_answer": result.get("top_answer", ""),
        "constraint_check_raw": constraint_check,
        "msce_elapsed": result.get("elapsed_time", 0),
    }


def main():
    print("=" * 70)
    print("MSCE Hubble Tension Analysis")
    print("6 Resolution Proposals × Multi-Constraint Consistency Check")
    print("=" * 70)
    print(f"\nProposals: {len(PROPOSALS)}")
    print(f"Constraints per proposal: 8 (CMB + BAO + SNe + BBN + S8 + Age + Gravity + Cross)")
    t_start = time.time()

    results = {}
    for key, info in PROPOSALS.items():
        try:
            results[key] = analyze_proposal(key, info)
            print(f"  MSCE conf={results[key]['msce_confidence']:.3f}, "
                  f"disag={results[key]['msce_disagreement']:.3f}, "
                  f"time={results[key]['msce_elapsed']:.1f}s")
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            results[key] = {"error": str(e)}

    total_time = time.time() - t_start

    # Summary table
    print("\n" + "=" * 70)
    print("HUBBLE TENSION ANALYSIS — SUMMARY")
    print("=" * 70)
    print(f"\n{'Proposal':<35} {'MSCE Conf':>10} {'Disag':>8} {'Time':>8}")
    print("-" * 65)
    for key, info in PROPOSALS.items():
        r = results.get(key, {})
        print(f"{info['name']:<35} {r.get('msce_confidence', 0):>10.3f} "
              f"{r.get('msce_disagreement', 0):>8.3f} {r.get('msce_elapsed', 0):>7.1f}s")

    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    output = {
        "timestamp": time.time(),
        "analysis_type": "hubble_tension_multi_constraint",
        "proposals": PROPOSALS,
        "results": {k: {
            "proposal_name": v.get("proposal_name", ""),
            "msce_confidence": v.get("msce_confidence", 0),
            "msce_disagreement": v.get("msce_disagreement", 0),
            "msce_top_answer": v.get("msce_top_answer", "")[:500],
            "constraint_check_raw": v.get("constraint_check_raw", "")[:500],
            "msce_elapsed": v.get("msce_elapsed", 0),
        } for k, v in results.items()},
        "summary": {
            "total_proposals": len(PROPOSALS),
            "total_time_s": round(total_time, 1),
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "results/hubble_tension_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
