# Constraint AI — 产品使用说明书

## 产品简介

Constraint AI 是一个幻觉检测 API。用户传入一段文本，API 返回这段文本有多大可能是 LLM 幻觉。

**核心原理：** 传统方法看模型"输出时是否自信"判断幻觉——但 LLM 在编造错误信息时往往跟说真话一样自信。我们看模型内部的"约束违反度"：当模型生成与训练知识不一致的内容时，隐藏状态会违反其内部约束，产生可检测的信号。**不管表面多自信，都能抓到。**

## 基准对比

TruthfulQA 上与其他方法的头对头对比（Qwen2.5-7B-Instruct）：

| 方法 | AUC | 为什么失败 |
|------|-----|-----------|
| **Constraint AI** | **0.816** | — |
| SelfCheckGPT | 0.250 | 三次输出一致错误 → 判定"安全" |
| 预测熵 | 0.465 | 自信地胡说 → 低熵 → 判定"安全" |
| 最大概率 | 0.447 | 高置信度 ≠ 正确 |

Cohen's d = 1.06（大效应量），p = 0.018，效果统计显著。召回率 100%（不漏检任何幻觉）。

## API 接口

### 基础地址

```
https://consequences-feed-assessed-cave.trycloudflare.com
```

### 认证

在请求头中携带 `X-API-Key`。试用请联系获取。

### 端点

#### 1. 单条检测 `POST /detect`

**请求：**
```json
{
  "text": "Do humans only use 10% of their brains?",
  "temperature": 0.6
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 待检测文本，最长 4096 字符 |
| temperature | float | 否 | 生成温度，默认 0.6，范围 0.0-2.0 |

**响应：**
```json
{
  "text": "Do humans only use 10% of their brains?",
  "hallucination_score": -0.0299,
  "hallucination_probability": 0.68,
  "risk_level": "high",
  "latency_ms": 1134,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "method": "constraint_residual"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| hallucination_score | float | 约束残差 Δ\|\|Π\|\|，越接近 0 = 越可能是幻觉 |
| hallucination_probability | float | 校准后的 P(幻觉)，0-1 |
| risk_level | string | low / medium / high / critical |
| latency_ms | float | 检测耗时（毫秒） |

**cURL 示例：**
```bash
curl -X POST https://consequences-feed-assessed-cave.trycloudflare.com/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"text": "Do vaccines cause autism?"}'
```

**Python 示例：**
```python
import requests

resp = requests.post(
    "https://consequences-feed-assessed-cave.trycloudflare.com/detect",
    headers={"Content-Type": "application/json", "X-API-Key": "YOUR_KEY"},
    json={"text": "Is the Great Wall of China visible from space?"},
)

result = resp.json()
print(f"P(hallucination) = {result['hallucination_probability']:.1%}")
print(f"Risk: {result['risk_level']}")
```

#### 2. 批量检测 `POST /detect/batch`

```json
[
  {"text": "Can sharks get cancer?"},
  {"text": "Is the Earth round?"}
]
```

返回每个的结果数组。

#### 3. Demo（免认证，限流） `POST /detect/demo`

无需 API Key，每 IP 限 5 次/分钟。适合快速体验。

```bash
curl -X POST https://consequences-feed-assessed-cave.trycloudflare.com/detect/demo \
  -H "Content-Type: application/json" \
  -d '{"text": "Does shaving make hair grow back thicker?"}'
```

#### 4. 健康检查 `GET /health`（公开）

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "cuda",
  "uptime_seconds": 3600,
  "requests_served": 128,
  "avg_latency_ms": 1123
}
```

#### 5. 用量查询 `GET /usage`（需认证）

```json
{
  "key_hash": "b2d132febc5c",
  "tier": "enterprise",
  "requests": 42,
  "monthly_limit": 999999
}
```

## 评分解读

### 怎么理解 hallucination_score 和 probability？

| 分数范围 | 概率范围 | 风险等级 | 含义 |
|----------|----------|----------|------|
| > -0.02 | > 72% | critical | 极可能包含幻觉 |
| -0.04 ~ -0.02 | 52-72% | high | 较可能包含幻觉 |
| -0.06 ~ -0.04 | 32-52% | medium | 不确定区域 |
| < -0.06 | < 32% | low | 大概率正确 |

分数越接近零 → 约束违反越大 → 幻觉可能性越高。分数越负 → 模型约束解析越强 → 事实可信度越高。

### 重要提示

- 概率来自 TruthfulQA 30 条样本的 logistic 校准，不保证每条都准确
- 检测延迟 ~1 秒，不适合实时对话场景
- 目前只支持英文文本
- 模型基于 Qwen2.5-7B-Instruct，不同 LLM 的幻觉检测能力可能不同

## 试用场景

| 场景 | 怎么用 |
|------|--------|
| 客服 AI 回复质检 | 把 AI 生成的回复逐条 POST 检测 |
| 内容生成审核 | 生成后 API 扫一遍，拦截高危内容 |
| 知识库 QA 校验 | 用户提问 → AI 回答 → API 验证 |
| 模型评估 | 批量跑数据集，对比不同模型幻觉率 |

## 套餐

| 套餐 | 价格 | 请求量 | 速率 |
|------|------|--------|------|
| Free | $0 | 1,000/月 | 10/分钟 |
| Pro | $199/月 | 50,000/月 | 60/分钟 |
| Enterprise | $999/月 | 无限 | 300/分钟 |

## 联系与支持

- API 试用 / 购买：联系获取 API Key
- 演示地址：https://consequences-feed-assessed-cave.trycloudflare.com
- 当前运行模型：Qwen2.5-7B-Instruct on CUDA GPU
