# StabilityBench arXiv/Zenodo/GitHub 提交清单

**日期：** 2026-05-17
**论文：** Recursive Stability Index (RSI, β)
**DOI：** 10.5281/zenodo.20041757

## arXiv 提交

### 准备
- [ ] LaTeX 模板准备（或提交 PDF + 源码）
- [ ] 标题：The Recursive Stability Index: A Benchmark for Multi-Generation LLM Degradation
- [ ] 摘要已更新（6 基线，困惑度 n=9 ρ=0.47）
- [ ] 作者：Deng Xinhang
- [ ] 分类：cs.CL (Computation and Language), cs.AI
- [ ] 交叉列表：stat.ML

### 数据附件
- [ ] all_models_summary.json — 16 模型 β 值
- [ ] 谱系数据（JSONL）— n100/ + latest_models/
- [ ] 跨种子比较数据
- [ ] 温度扫描数据
- [ ] K=5 扩展数据
- [ ] 困惑度基线数据
- [ ] P3 神经验证数据
- [ ] 纯 judge 提取器 v4 结果

## Zenodo 更新

- [ ] 上传最新论文 PDF（EN + CN）
- [ ] 上传完整实验数据
- [ ] 上传分析脚本
- [ ] 更新 DOI 记录描述
- [ ] 版本号：v2（包含 6 基线）

## GitHub

- [ ] 创建公开仓库
- [ ] README.md：项目概述 + 快速开始
- [ ] LICENSE（建议 CC-BY-4.0 或 MIT）
- [ ] .gitignore（排除 .env、__pycache__、.claude/）
- [ ] 目录结构文档

### 仓库结构
```
stabilitybench/
├── README.md
├── paper/
│   ├── main.md / main.pdf          # EN 论文
│   └── main_cn.md / main_cn.pdf   # CN 论文
├── synthetic_decay_monitor/        # 核心库
├── experiment_data/                # 实验数据
├── run_*.py                        # 分析脚本
├── requirements.txt
└── LICENSE
```

## 提交前终检

- [ ] 所有 β 值与源数据一致
- [ ] 16 模型榜单完整
- [ ] 6 基线结果全部记录
- [ ] 跨种子源验证数据清洁
- [ ] 温度稳健性数据完整
- [ ] PDF 构建无错误
- [ ] 中英文版本内容同步
- [ ] 所有引用格式正确
