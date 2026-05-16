# K=5 Extension: GPT-4o-mini + Claude Opus 4.7

## Setup Required


To extend K=5 validation beyond DeepSeek family, add these 2 model entries to `run_k5_experiment.py` line 27-31 (MODELS list):

```python
{"model": "gpt-4o-mini",      "provider": "quickrouter", "family": "OpenAI",
 "name": "gpt-4o-mini_k5",     "temp": 0.8, "delay": 3.0},
{"model": "claude-opus-4-7",   "provider": "quickrouter", "family": "Anthropic",
 "name": "claude-opus-4-7_k5",  "temp": 0.8, "delay": 3.0},
```

Then add to `analyze_k5.py` line 29 (MODEL_NAMES list):
```python
"gpt-4o-mini_k5",
"claude-opus-4-7_k5",
```

## Cost
- 100 seeds × 5 gens × 2 models = 1000 QuickRouter calls
- ~$10-15, ~2-3 hours runtime

## Scientific Value
- Current K=5 evidence: 21/21 exponential wins, but ALL from DeepSeek family
- Adding GPT-4o-mini (mid-β, 0.088) and Claude Opus 4.7 (high-β, 0.164) tests exponential model across different architectures and β ranges
- Would strengthen "exponential model is architecture-independent" claim
