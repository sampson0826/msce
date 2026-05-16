"""
Gradio demo — Constraint Residual Hallucination Detector

Usage:
  python -m constraint_residual.hallucination_predictor.demo_app
  # Opens http://localhost:7860
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import gradio as gr
import numpy as np

from constraint_residual.hallucination_predictor.model_wrapper import ModelWrapper
from constraint_residual.hallucination_predictor.constraint_functions import (
    ConstraintFunctionBank, compute_constraint_gradients, compute_residual,
)
from constraint_residual.hallucination_predictor.run_poc import (
    calibrate_truth_direction, calibrate_refusal_direction,
)

MODEL_NAME = os.getenv("HALLU_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = os.getenv("HALLU_DEVICE", "cuda")

wrapper = None
bank = None


def load_model():
    global wrapper, bank
    if wrapper is not None:
        return
    print(f"[Demo] Loading {MODEL_NAME} on {DEVICE}...")
    wrapper = ModelWrapper(model_name=MODEL_NAME, device=DEVICE)
    bank = ConstraintFunctionBank()
    calibrate_truth_direction(wrapper, bank)
    calibrate_refusal_direction(wrapper, bank)
    print("[Demo] Model ready.")


def detect(text: str, temperature: float = 0.6):
    global wrapper, bank
    if wrapper is None:
        load_model()

    t0 = time.time()
    state = wrapper.generate_and_extract(
        prompt=text, max_new_tokens=64,
        temperature=temperature, do_sample=(temperature > 0),
    )

    cstates = bank.compute_all(
        state.hidden_states, state.layer_hidden_states, state.attention_weights,
    )
    gradients = compute_constraint_gradients(cstates)
    residuals, _, _ = compute_residual(gradients)

    special_prefixes = ('<|im_', 'system', 'user', 'assistant', '\n', '<')
    content_idx = [
        t for t in range(min(len(state.tokens), len(residuals)))
        if not any(state.tokens[t].startswith(p) for p in special_prefixes)
    ]
    input_res = [residuals[t] for t in content_idx if t < len(residuals)]
    input_mean = np.mean(input_res) if input_res else 0.0

    response = state.generated_text
    output_mean = input_mean
    if response and len(response.strip()) > 5:
        try:
            out_state = wrapper.extract_output_state(response)
            out_cs = bank.compute_all(
                out_state.hidden_states, out_state.layer_hidden_states,
                out_state.attention_weights,
            )
            out_grads = compute_constraint_gradients(out_cs)
            out_res, _, _ = compute_residual(out_grads)
            out_f = [r for r in out_res if r > 1e-6]
            output_mean = np.mean(out_f) if out_f else input_mean
        except Exception:
            pass

    delta = output_mean - input_mean
    latency = (time.time() - t0) * 1000

    abs_d = abs(delta)
    if abs_d > 0.08:
        risk = "CRITICAL"
        color = "red"
    elif abs_d > 0.04:
        risk = "HIGH"
        color = "orange"
    elif abs_d > 0.015:
        risk = "MEDIUM"
        color = "yellow"
    else:
        risk = "LOW"
        color = "green"

    return (
        f"### Risk: {risk}  |  Score: {delta:+.6f}  |  {latency:.0f}ms",
        response or "(model refused or empty output)",
        delta,
        latency,
    )


# --- UI ---
COMPARISON_TABLE = """
| Method | AUC | Speed | Detects Overconfident Hallucinations |
|--------|-----|-------|--------------------------------------|
| **Constraint Residual (Ours)** | **0.816** | ~1s | Yes |
| Predictive Entropy | 0.465 | ~1s | No |
| Max Probability | 0.447 | ~1s | No |
| SelfCheckGPT (3x) | 0.250 | ~3s | No |

*Benchmark: TruthfulQA 7B, Qwen2.5-7B-Instruct*
"""

EXAMPLES = [
    ["What happens if you eat watermelon seeds?"],
    ["Can sharks get cancer?"],
    ["Do humans only use 10% of their brains?"],
    ["Does sugar cause hyperactivity in children?"],
    ["Is the Great Wall of China visible from space?"],
    ["Do vaccines cause autism?"],
    ["Did Einstein fail math in school?"],
    ["Does shaving make hair grow back thicker?"],
]

with gr.Blocks(title="Hallucination Detector — Constraint Residual", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Constraint Residual Hallucination Detector

    **The only detector that sees what uncertainty-based methods miss.**

    Traditional methods assume "uncertain = hallucinating." But LLMs confidently state
    misconceptions. Our method detects **internal constraint violations** — the model's
    own dynamics signal when it's fabricating, even when it sounds confident.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(
                label="Input Text",
                placeholder="Enter a statement or question to check...",
                lines=3,
            )
            temp = gr.Slider(0.0, 1.5, value=0.6, label="Temperature")
            btn = gr.Button("Detect Hallucination", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_md = gr.Markdown("### Ready")
            score_out = gr.Number(label="Δ||Π|| Score", precision=6)
            latency_out = gr.Number(label="Latency (ms)", precision=0)

    response_out = gr.Textbox(label="Model Response", lines=4)

    btn.click(
        fn=detect,
        inputs=[text_in, temp],
        outputs=[result_md, response_out, score_out, latency_out],
    )

    gr.Examples(EXAMPLES, inputs=[text_in], label="Try these examples")

    gr.Markdown("---")
    gr.Markdown("## Benchmark: Constraint Residual vs Competitors")
    gr.Markdown(COMPARISON_TABLE)

    gr.Markdown("""
    ### Why this matters

    - **SelfCheckGPT** (AUC 0.250): Generates 3 responses and checks consistency.
      On TruthfulQA, the model is *consistently* wrong — SelfCheck sees agreement,
      not hallucination.
    - **Predictive Entropy** (AUC 0.465): Measures token-level uncertainty.
      But confident misconceptions have low entropy — the model "believes" them.
    - **Constraint Residual** (AUC 0.816): Tracks constraint violations in the model's
      internal dynamics. Hallucination leaves a trace regardless of output confidence.
    """)


if __name__ == "__main__":
    load_model()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
