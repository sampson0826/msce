---
title: Synthetic Data Decay Monitor
emoji: 📉
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5"
app_file: app.py
pinned: false
license: mit
---

# Synthetic Data Decay Monitor

**Constraint-layer health diagnostic for AI training pipelines.** No GPU required.

When an LLM generates synthetic data used to train the next generation, output quality degrades — like repeatedly photocopying a document. This tool detects **which executor type** is failing, **how fast** each capability is decaying, and **what data** is needed to fix it — before model collapse happens.

## How to Deploy

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** SDK
3. Clone the Space and copy these files into it
4. Push to HuggingFace

## Local Development

```bash
pip install -r requirements.txt
python app.py
```
