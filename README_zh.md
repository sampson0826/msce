# MSCE — 多源一致性引擎

**科学声明的系统性交叉校验基础设施。**

[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

## 这是什么？

物理学家提出一个解决哈勃张力的新理论时，通常只验证 1-2 项观测条件。但实际上有**8项独立校验条件必须同时满足**。MSCE 一次性全量并行检查——发现单审查者无法看到的结构性条件冲突。

MSCE 不是 AI 模型，是**多源验证系统**。它不生成答案，它检测"这些声明能否在逻辑上同时成立"。

> **MSCE 之于验证，如同编译器之于代码。** 编译器不写程序——它检查程序能否运行。MSCE 不提理论——它检查理论能否同时满足它所声称的所有校验条件。

## 快速演示

```bash
git clone https://github.com/sampson0826/msce.git
cd msce
pip install -e .
msce check hubble --quick
```

**输出：** 6 个主流 H₀ 方案 × 8 项独立校验条件的交叉验证矩阵。**全部标红。**

## 哈勃张力核心结果

| 方案 | 通过 | 违反 | MSCE 置信度 |
|------|------|------|-----------|
| 早期暗能量 (EDE) | 3 | 3 | **0.076** |
| 修改引力 (f(R)) | 3 | 4 | 0.253 |
| 额外中微子 (ΔN_eff) | 3 | 2 | 0.287 |
| 衰变暗物质 (DDM) | 5 | 2 | 0.358 |
| 局部空洞假说 | 6 | 2 | 0.171 |
| 未知系统误差 | 6 | 0 | 0.108 |

**双因子组合比单方案更差**——物理机制非线性相互作用，产生新冲突而非解决旧冲突。

## Benchmark：206 题评测

MSCE 在 206 道跨领域验证任务中达到 **87.4% 准确率**，GPT-5.5 仅 74.8%——提升 **+12.6 个百分点**。

| 领域 | GPT-5.5 | MSCE | 提升 |
|------|---------|------|------|
| **跨域综合** | 54.5% | **84.9%** | **+30.3%** |
| **科学** | 73.0% | **97.3%** | **+24.3%** |
| 条件依赖分析 | 55.8% | 67.4% | +11.6% |
| 逻辑 | 85.2% | 92.6% | +7.4% |
| 数学 | 93.3% | 96.7% | +3.3% |
| 语言 | 94.4% | 91.7% | -2.8% |

MSCE 在验证密集型领域显著领先，在开放创意任务中略微落后——**这正是验证系统应有的保守性。**

## 核心差异：校准不确定性

GPT-5.5 在 40 个案例中给出高度自信（>0.8）的错误答案。MSCE 的平均置信度仅 0.49，却在准确率上高出 12.6 个百分点。在高风险验证场景中，诚实的"不确定"远比自信的"错误"有价值。

## 安装

```bash
git clone https://github.com/sampson0826/msce.git
cd msce
pip install -e .
```

需要 Python 3.10+，无需 GPU。可视化功能：`pip install -e ".[notebook]"`

## 引用

```bibtex
@software{msce2026,
  title={MSCE: Multi-Source Consistency Engine},
  author={Deng, Xinhang and MSCE Collaboration},
  year={2026},
  doi={10.5281/zenodo.20041757},
  url={https://github.com/sampson0826/msce}
}
```
