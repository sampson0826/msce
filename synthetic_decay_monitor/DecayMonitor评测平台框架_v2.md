# DecayMonitor — LLM 递归稳定性评测平台 v2

**2026-05-12 | β 单维度版**

---

## 第一部分：平台运行框架

### 1.1 平台定位

在 LLM 评测矩阵中新增**递归稳定性 β** 这个维度。

```
现有评测:
  MMLU       → 知识覆盖面
  HumanEval  → 代码生成
  Chatbot Arena → 人类偏好
  HELM       → 多维度综合

DecayMonitor:
  β (递归稳定性) → "模型在自主循环中能撑多久"
```

现有评测全部测单次输入→输出质量。β 测的是模型反复读取自己输出后的退化速率。对 AI agent 开发者，这个维度直接影响"选哪个模型跑我的 agent"。

### 1.2 平台运行流程

```
┌─────────────────────────────────────────────────────────┐
│                DecayMonitor 运行流程                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  每周一 06:00 UTC                                        │
│     │                                                   │
│     ├─ Celery Beat 触发定时任务                           │
│     │                                                   │
│     ├─ 读取 models_registry.json（待评测模型列表）         │
│     │                                                   │
│     ├─ 逐个模型:                                         │
│     │   ├─ ProviderAdapter.generate() × 12 seeds × 3 gens│
│     │   ├─ ext_text_features() → 8 特征 (CPU only)       │
│     │   ├─ text_features_to_constraint() → S_n × 5 维度  │
│     │   ├─ DecayEngine → β × 6 能力维度                  │
│     │   └─ ExecutorClassifier → E-I/E-II/E-III 退化诊断   │
│     │                                                   │
│     ├─ 结果写入 PostgreSQL + JSONL 归档                   │
│     └─ Redis 缓存更新 → 前端排行榜刷新                     │
│                                                         │
│  新模型上线: 即时触发评测，48h 内上架                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 模型准入管理

**models_registry.json：**

```json
{
  "models": [
    {
      "id": "claude-sonnet-4-6",
      "name": "Claude Sonnet 4.6",
      "family": "Claude",
      "provider": "quickrouter",
      "status": "active",
      "added": "2026-05-01",
      "last_evaluated": "2026-05-12",
      "beta": 0.3106,
      "tags": ["commercial", "text-only"]
    }
  ]
}
```

**准入规则：**
- 模型必须可通过公开 API 访问（保证评测可复现）
- 开源模型需提供 HuggingFace 链接 + 推理配置
- 模型更新（同 ID 新版本）自动触发复测

### 1.4 排行榜设计

**首页 = β 总榜：**

```
排名  模型                β↓  数学  代码  事实  逻辑  创意  通用  状态
───  ─────────────────  ───  ────  ────  ────  ────  ────  ────  ────
 1   Qwen2.5-7B        0.29  ●    ●    ●    ●    ●    ●   stable
 2   Claude Opus 4.6    0.30  ●    ●    ●    ●    ●    ●   stable
 3   Claude Sonnet 4.6  0.31  ●    ●    ●    ●    ●    ●   stable
 4   Claude Haiku 4.5   0.31  ●    ●    ●    ●    ●    ●   stable
...

颜色: ● 优秀(<0.20)  ● 良好(<0.30)  ● 一般(<0.35)  ● 差(>0.35)
β 越低越好。
```

**模型详情页：**
- 6 维衰减曲线（交互式）
- S_n 跨代变化轨迹
- 执行者退化类型诊断（E-I/E-II/E-III）
- 同家族模型 β 对比
- 原始 JSONL 数据下载

**模型对比页：**
- 选 2-4 模型，并排对比 β × 6 能力
- 自动标注：模型 A 在数学推理上比模型 B 稳定 X%

### 1.5 质量保证

| 机制 | 频率 | 说明 |
|------|:----:|------|
| 评测复现性检查 | 每模型每次 | 同模型两次评测 β 偏差 < 0.03 |
| β 历史趋势监控 | 周度 | 同模型 β 突然跳变 → 人工复核 |
| 模型更新复测 | 即时 | 模型厂商通知更新后 48h 内复测 |
| 社区反馈通道 | 持续 | GitHub Issue 接收模型厂商申诉 |

---

## 第二部分：商业框架

### 2.1 价值主张

> 现有的 LLM 评测告诉你模型"答对多少题"。
> **β 告诉你模型在自主循环中能撑多久。**

对于 AI agent 开发者：MMLU 第一的模型，如果 β=0.35，10 轮后约束质量只剩 1.3%。β 是 benchmark 测不到的决策维度。

### 2.2 收入来源（三档）

| | Free | Pro | Enterprise |
|------|:---:|:---:|:---:|
| **价格** | $0 | $1,499/次 | $4,999/月 |
| **seeds × gens** | 12 × 3 | 100 × 5 | 自定义 |
| **结果公开** | 上榜 | **闭门（不上榜）** | **闭门** |
| **执行者诊断** | 基本 | 深度 E-I/E-II/E-III | 深度 + 干预建议 |
| **竞品对比** | 公开数据 | 匿名对比 | 指定竞品对比 |
| **API 访问** | 50 次/天 | 包含 | 无限 |
| **历史 β 追踪** | 否 | 否 | ✅ 每次 checkpoint 曲线 |
| **训练数据诊断** | 否 | 否 | ✅ β 跨能力差异 → 数据盲区 |
| **交付物** | — | PDF + CSV/JSONL | PDF + API + Slack |
| **目标客户** | 所有人 | 模型厂商发布前 | 训练团队每 checkpoint |

**定价逻辑：** LMArena 定制评测 $5,000-50,000/次作为参照。$1,499 是初期获客价，首 10 个客户后上调至 $2,999。

### 2.3 目标客户画像

**第一优先级：模型厂商（发布前评测）**
- 人物：LLM 厂商的 evaluation 负责人
- 场景：新版本发布前，需要 benchmark 之外的独立数据
- 痛点：beta 用户反馈 agent 场景表现不稳定，找不到原因
- 我们提供：β 作为发布前 checklist 的新维度——没有人测这个
- 获客：论文发表后定向联系 Anthropic/DeepSeek/01.AI 的 eval 团队

**第二优先级：AI agent 初创公司（选型决策）**
- 人物：CTO / 技术负责人
- 场景：选模型跑 agent——GPT-4o 还是 Claude Sonnet
- 痛点：benchmark 数据没区分度，实际 agent 表现跟 benchmark 不一致
- 我们提供：β 排名 + 场景化对比 + "选型指南"内容
- 获客：内容营销 + HuggingFace demo

**第三优先级：AI 投资/研究机构**
- 人物：分析师
- 场景：需要客观的模型稳定性量化指标
- 我们提供：β 历史趋势 + 跨家族 β 对比

### 2.4 收入预测

| 时间 | 模型数 | Pro/月 | Enterprise | 月收入 |
|------|:----:|:----:|:----:|:----:|
| Month 1-3 | 4→10 | 0 | 0 | $0 |
| Month 4-6 | 15 | 2 | 0 | $3,000 |
| Month 7-9 | 20 | 4 | 1 | $11,000 |
| Month 10-12 | 25 | 7 | 2 | $20,500 |
| Month 13-18 | 35 | 10 | 4 | $35,000 |
| Month 19-24 | 50 | 15 | 6 | $52,500 |

### 2.5 成本结构

**月运营成本：**

| 项目 | 月费 |
|------|:----:|
| 云服务器（4 vCPU + 8GB） | $60 |
| PostgreSQL managed | $30 |
| Redis | $15 |
| 域名 + CDN | $20 |
| β 评测 API 费用（30 模型 × $0.50） | $15 |
| **合计** | **$140** |

**盈亏平衡：月收入 > $140。即每月 1 个 Pro 客户。**

### 2.6 竞争壁垒

| 壁垒 | 性质 | 持久性 |
|------|------|:----:|
| β 定义权 + 约束吸引子坍缩理论 | 学术 + 品牌 | 长期 |
| CPU-only 评测（$0.50/模型，零门槛） | 技术 | 中长期 |
| 跨家族 β 数据库（先发优势） | 数据 | 中长期 |
| 多 Provider 抽象层（快速接入新模型） | 工程 | 短期 |
| 独立第三方立场（不收模型厂商赞助） | 品牌 | 长期 |

**最难复制的：独立第三方信任。** 不收模型厂商赞助，排行榜排序完全独立。

### 2.7 退出策略

| 路径 | 概率 | 预期 |
|------|:----:|------|
| 被 LMArena/HuggingFace/Weights&Biases 收购 | 中 | $3-10M |
| 独立运营，小团队自给自足 | 高 | 2-3 人，$30-50k MRR |
| 融资规模化 | 低 | 需证明 TAM > $50M |

---

## 第三部分：技术实现框架

### 3.1 系统架构总览

```
                          decaymonitor.ai
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
      CDN (静态资源)      API Gateway          Task Queue
      React SPA          FastAPI + Nginx      Celery + Redis
           │                   │                   │
           │              ┌────┴────┐              │
           │              │         │         β Worker
           │          Auth Svc   Eval Svc    (Celery)
           │          (JWT)     (REST)           │
           │              │         │              │
           └──────────────┴─────────┴──────────────┘
                                     │
                           PostgreSQL + S3/MinIO
                           
评测引擎:
  ProviderAdapter → ExperimentRunner → DecayEngine → ExecutorClassifier
```

### 3.2 β 评测算法（三步）

#### 第一步：文本 → 8 特征（纯规则，CPU only）

```
输入文本
    │
    ├─ E-I（逻辑层）
    │   ├─ ei_logic_density     = count(逻辑连接词) / n_tokens
    │   └─ ei_syntax_cv          = std(句子长度) / mean(句子长度)
    │
    ├─ E-II（风格层）
    │   ├─ eii_bigram_repetition = count(重复bigram) / total_bigrams
    │   ├─ eii_unique_word_ratio  = len(set(words)) / len(words)
    │   ├─ eii_filler_ratio       = count(填充词) / n_words
    │   └─ eii_truncation_ratio   = count(截断长词) / count(长词)
    │
    └─ E-III（边界层）
        ├─ eiii_proper_case_ratio = count(首字母大写词) / n_words
        └─ eiii_number_integrity  = count(未变数字) / count(数字)
```

纯规则提取，中英文自动检测，长文本自适应分块（80-200 tokens Goldilocks 区）。

**已验证：** E-I 与 LLM judge 相关性 0.93，E-II 相关性 0.98，E-III 相关性 0.27（较弱，权重已降低）。

#### 第二步：8 特征 → 约束状态 S_n

```
sigma_syntax  = 0.65×ei_logic + 0.35×ei_syntax_cv
sigma_style   = unique_ratio×(1 - 0.5×rep - 0.3×trunc - 0.2×filler)
sigma_fact    = 0.55×proper_case + 0.45×number_integrity
sigma_coherence = 0.55×ei_logic + 0.45×(1 - rep)
sigma_safety  = 0.5

S_n = (σ_fact + σ_syntax + σ_style + σ_safety + σ_coherence) / 5
```

#### 第三步：S_n 序列 → β

```
指数衰减模型: S_n = S_0 × (1-β)^n
对数线性化:   log(S_n/S_0) = n × log(1-β)

最小二乘线性回归 → 斜率 k = log(1-β) → β = 1-e^k

6 能力维度各算一个 β → 全局 β = mean(β_math, β_code, β_fact, β_logic, β_creative, β_general)
```

#### 状态分级

| S_n | 状态 | |
|------|------|------|
| > 0.80 | healthy | 约束结构完整 |
| 0.50-0.80 | degrading | 开始退化 |
| 0.30-0.50 | critical | 接近坍缩 |
| < 0.30 | collapsed | 已坍缩 |

### 3.3 数据库设计

```sql
-- 模型注册表
CREATE TABLE models (
    id              VARCHAR(100) PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    family          VARCHAR(50),
    provider        VARCHAR(50) NOT NULL,
    status          VARCHAR(20) DEFAULT 'active',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- β 评测结果
CREATE TABLE beta_results (
    id              SERIAL PRIMARY KEY,
    model_id        VARCHAR(100) REFERENCES models(id),
    evaluated_at    TIMESTAMP DEFAULT NOW(),
    global_beta     FLOAT NOT NULL,
    beta_math       FLOAT,
    beta_code       FLOAT,
    beta_fact       FLOAT,
    beta_logic      FLOAT,
    beta_creative   FLOAT,
    beta_general    FLOAT,
    n_seeds         INT DEFAULT 12,
    n_generations   INT DEFAULT 3,
    lineage_path    VARCHAR(500),
    diagnosis       JSONB,
    meta            JSONB
);

-- 用户表
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE,
    name            VARCHAR(100),
    avatar_url      VARCHAR(500),
    auth_provider   VARCHAR(20),       -- 'github', 'google', 'email'
    auth_id         VARCHAR(100),
    plan            VARCHAR(20) DEFAULT 'free',
    pro_credits     INT DEFAULT 0,
    stripe_customer_id VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 评测请求（Pro 用户）
CREATE TABLE eval_requests (
    id              SERIAL PRIMARY KEY,
    user_id         INT REFERENCES users(id),
    model_name      VARCHAR(200),
    status          VARCHAR(20) DEFAULT 'pending',
    result_beta     FLOAT,
    report_path     VARCHAR(500),
    requested_at    TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP
);
```

### 3.4 API 设计

```
公开 API (无需认证):
  GET  /api/v1/leaderboard?sort=beta&limit=20
  GET  /api/v1/models/{model_id}
  GET  /api/v1/models/{model_id}/beta-history
  POST /api/v1/compare  { models: [...], dimensions: [...] }

认证 API (需 JWT):
  POST /api/v1/auth/github     { code }
  POST /api/v1/auth/google     { code }
  POST /api/v1/auth/register   { email, password }

Pro API (需 Pro+ 订阅):
  POST /api/v1/eval/private    { model, seeds, generations }
  GET  /api/v1/eval/status/{request_id}
  GET  /api/v1/eval/download/{request_id}

Enterprise API (需 Enterprise 订阅):
  POST /api/v1/eval/continuous { model, schedule, webhook_url }
  GET  /api/v1/enterprise/dashboard
```

### 3.5 技术栈

| 层 | 技术选型 | 理由 |
|------|------|------|
| 前端 | React + Next.js + Tailwind | 生态成熟，SEO 友好（SSR） |
| 图表 | ECharts | 6 维雷达图 + 衰减曲线，中文友好 |
| 后端 | FastAPI (Python) | 与评测引擎同语言，零桥接 |
| 任务队列 | Celery + Redis | Python 原生 |
| 数据库 | PostgreSQL | JSONB 支持，评测数据结构灵活 |
| 文件存储 | S3 / MinIO | JSONL 归档 + 报告 PDF |
| 认证 | NextAuth.js + python-jose | GitHub OAuth 一行配置 |
| 支付 | Stripe | 标准方案 |
| 部署 | Docker + GitHub Actions CI | 不锁定云厂商 |

### 3.6 开发阶段

**Phase D-1: 网站 MVP（4 周）**

| 周 | 任务 |
|:----:|------|
| W1 | FastAPI 骨架 + 数据库 schema + 评测引擎对接 |
| W2 | 排行榜首页 + 模型详情页 + 图表 |
| W3 | 模型对比页 + 交互优化 |
| W4 | Celery 定时评测 + 认证 + 部署上线 |

**Phase D-2: 付费功能（4 周）**

| 周 | 任务 |
|:----:|------|
| W1 | Pro 评测流程 + 报告生成 |
| W2 | Stripe 集成 + 权限控制 |
| W3 | Enterprise dashboard + 历史追踪 |
| W4 | 测试 + 灰度上线 |

---

## 第四部分：术语定义

| 术语 | 定义 |
|------|------|
| **β** | 递归衰减率。模型每代丢失的约束质量比例。∈ [0, 1]，越低越好。 |
| **S_n** | 第 n 代的约束基完整性。5 维 σ 的均值。∈ [0, 1]。 |
| **约束吸引子坍缩** | C_div = std(∥Π∥) 随代数指数衰减的现象。∥Π∥ 均值稳定但方差坍缩 → 模型在少数模式上"锁死"。 |
| **E-I / E-II / E-III** | 三种执行者退化类型。E-I=逻辑断裂(α=0.40)，E-II=风格均质化(α=0.20)，E-III=事实侵蚀(α=0.08)。 |
| **递归生成** | 模型将前一代输出作为下一代的输入。模拟 agent loop 中的退化。 |

---

## 第五部分：当前状态与启动条件

### 已完成

| 资产 | 状态 |
|------|:----:|
| β 算法 + 4 模型数据库 | ✅ |
| ExperimentRunner 统一评测框架 | ✅ |
| ProviderAdapter 多后端 | ✅ |
| 纯文本特征提取器（CPU only） | ✅ |
| 产品方向确认（评测平台） | ✅ |
| 商业模型框架 | ✅ |

### 待启动

| 优先级 | 任务 | 前置条件 | 费用 |
|:----:|------|------|:----:|
| P0 | DeepSeek-V3 β 实验（第三家族） | API key | $5 |
| P0 | P3 严格检验（逐维 π 分量） | GPU 机器 | $0 |
| P1 | n=100 seeds 统计验证 | API 费用 | $126 |
| P1 | Llama 3 β 实验（第四家族） | Together API | $10 |
| P2 | 论文初稿 | P0+P1 数据 | $0 |
| P3 | 网站 MVP 开发 | 论文提交后 | 自有时间 |
| P3 | 付费功能开发 | 有 Pro 意向客户后 | 自有时间 |

**总费用 < $150。唯一瓶颈：GPU（P3 严格检验）。**
