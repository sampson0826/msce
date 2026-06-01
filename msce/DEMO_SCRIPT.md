# MSCE Demo Video Script (2 minutes)

---

## Segment 1 — Title Card [0:00-0:10]

**Visual:** Black background. Text fades in.

> MSCE — Know When to Trust AI

Subtitle: "6 AI models. One truth."

---

## Segment 2 — The Problem [0:10-0:25]

**Visual:** Screen recording of GPT-4o chat interface.

**Action:**
1. Type a math problem into GPT-4o: "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What is the average speed?"
2. GPT-4o responds: "50 km/h" (wrong — correct answer is 48 km/h)
3. Highlight the answer with a red underline.

**Narration (voiceover):**
"Single AI models are confidently wrong 30% of the time. And they can never tell you when they don't know."

**Visual cue:** Red "X" overlay on the wrong answer.

---

## Segment 3 — MSCE in Action [0:25-0:45]

**Visual:** Split screen. Left: MSCE terminal/dashboard. Right: same problem.

**Action:**
1. Same problem fed into MSCE via CLI or API.
2. Show 6 cognitive strategy labels appearing one by one:
   - `deep_first` → answer
   - `breadth_first` → answer
   - `counterfactual` → answer
   - `direct` → answer
   - `science_deep` → answer
   - `constraint_propagation` → answer
3. Adversarial elimination: answers clash, judge eliminates inconsistent ones.
4. Final verdict appears: **48 km/h** (green highlight).
5. Confidence score: **High (92%)**.

**Narration:**
"MSCE runs 6 heterogeneous cognitive strategies against the same problem. An adversarial elimination tournament finds the answer — and tells you how confident it is."

---

## Segment 4 — Uncertainty Quantification [0:45-1:05]

**Visual:** New problem input.

**Action:**
1. Type an intentionally ambiguous/edge-case problem.
2. The 6 models generate divergent answers.
3. Show the "disagreement map" — a visual showing 4 different answers from 6 models.
4. Final output: **Low confidence** warning in yellow/orange.
5. MSCE output: `confidence: low`, `verdict: "Models disagree. Recommend human review."`

**Narration:**
"When models disagree, MSCE tells you. This is the real innovation — an AI that knows when it doesn't know. Single models can't do this."

**Visual cue:** Yellow warning triangle with "Low Confidence — Verify Manually" text.

---

## Segment 5 — Benchmark Results [1:05-1:25]

**Visual:** Animated bar chart.

```
Chinese Math Benchmark (20 questions)
MSCE          ████████████████████ 95.0%
DeepSeek-V3   ██████████████████   90.0%
GPT-4o        ██████████████       70.0%

MMLU Pilot (30 questions)
MSCE          █████████████████░░ 86.7%
DeepSeek-V3   ████████████████░░░ 83.3%
GPT-4o        ███████████░░░░░░░░░ 56.7%
```

**Narration:**
"On Chinese math: 95% vs GPT-4o's 70%. On MMLU: 86.7% vs 56.7%. Not just more accurate — fundamentally more honest."

---

## Segment 6 — Quick Start [1:25-1:45]

**Visual:** Terminal screen recording. Fast-paced.

**Action:**
```
$ git clone https://github.com/.../msce
$ cd msce
$ pip install -r requirements.txt
$ export OPENAI_API_KEY=sk-...
$ python run_benchmark.py
```

Show output flowing. Benchmark completes. Results displayed.

**Narration:**
"Open source. MIT license. You bring your own API keys. Clone, install, run."

**Visual cue:** GitHub star button animation in corner.

---

## Segment 7 — Closing [1:45-2:00]

**Visual:** Black background. Logo. Text.

> MSCE
> Know When to Trust AI

> github.com/.../msce
> MIT License

**Narration:**
"MSCE — open source cognitive adversarial engine. Star us on GitHub. Link below."

**Visual cue:** Fade to black.

---

## Production Notes

- **Total runtime:** 2:00 exactly
- **Music:** Subtle, tech-forward instrumental. Not distracting.
- **Text overlays:** Use clean sans-serif font (Inter, SF Pro, or similar).
- **Screen recording resolution:** 1920x1080 minimum.
- **Code font:** Monospace (JetBrains Mono or Fira Code).
- **Color palette:** Dark theme. Green = correct, red = wrong, yellow/orange = uncertain.
- **Captions:** Burned-in English captions for all voiceover segments (accessibility + social media auto-play).
- **End screen:** GitHub link + star button animation for the last 5 seconds.
