# MSCE — 多源约束引擎

**检测科学理论与所引用数据之间的隐藏冲突。**

[![PyPI](https://img.shields.io/badge/pip%20install-msce-blue)](https://pypi.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

## 这是什么？

物理学家提出一个解决哈勃张力的新理论时，通常只检查1-2个约束。但实际上有**8项独立观测约束必须同时满足**。MSCE一次性全量检查——发现单个审稿人无法看到的跨约束冲突。

## 30秒演示

```bash
pip install msce
msce check hubble --quick
```

**输出：** 6个主流H₀方案 × 8项约束的热力图。**全部红色。**

## 哈勃张力核心结果

| 方案 | 通过 | 冲突 | MSCE置信度 |
|------|------|------|-----------|
| 早期暗能量 (EDE) | 3 | 3 | **0.076** |
| 修改引力 (f(R)) | 3 | 4 | 0.253 |
| 额外中微子 (ΔN_eff) | 3 | 2 | 0.287 |
| 衰变暗物质 (DDM) | 5 | 2 | 0.358 |
| 局部空洞 | 6 | 2 | 0.171 |
| 系统误差 | 6 | 0 | 0.108 |

**双因子组合比单方案更差**——物理机制非线性相互作用，产生新冲突而非解决旧冲突。

## Benchmark：206题评测

MSCE在206道约束密集型题目上达到**87.4%准确率**，GPT-5.5仅74.8%——提升**+12.6%**。

| 领域 | GPT-5.5 | MSCE | 提升 |
|------|---------|------|------|
| 跨域综合 | 54.5% | **84.9%** | **+30.3%** |
| 科学 | 73.0% | **97.3%** | **+24.3%** |
| 约束传播 | 55.8% | 67.4% | +11.6% |
| 逻辑 | 85.2% | 92.6% | +7.4% |
| 数学 | 93.3% | 96.7% | +3.3% |
| 语言 | 94.4% | 91.7% | -2.8% |

MSCE在约束密集领域显著领先，在开放创意任务中略微落后——**这正是设计的边界。**

## 安装

```bash
pip install msce
```

需要 Python 3.10+，无需GPU。

## 引用

```bibtex
@software{msce2026,
  title={MSCE: Multi-Source Constraint Engine},
  author={Deng, Xinhang and MSCE Collaboration},
  year={2026},
  doi={10.5281/zenodo.20041757},
  url={https://github.com/msce-ai/msce}
}
```
