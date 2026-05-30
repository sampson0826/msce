# HN Post Draft

**Title:** Show HN: MSCE — All 6 Hubble tension solutions fail cross-constraint checks (here's the data)

**Body:**

Every physicist I know has the same complaint about peer review: no single reviewer can check all the observational constraints a theory must satisfy.

So I built a tool that does exactly that. It's called MSCE — a constraint conflict detector.

**What it does:** Takes a scientific theory and checks it against all known observational constraints simultaneously. If the theory satisfies constraint A but violates constraint B, MSCE flags the conflict.

**How I tested it:** The Hubble tension is perfect for this. There are 6 mainstream solutions, and 8 independent observational constraints they must ALL satisfy (CMB power spectrum, BAO scale, supernova Hubble diagram, BBN abundances, S₈ tension, universe age, gravity tests, and cross-constraint consistency).

**The result:** None pass. All 6 proposals have MSCE confidence < 0.36. The "best" candidate (Decaying Dark Matter) scores 0.358.

Even more interesting: 2-factor combinations perform WORSE than single proposals. Because the mechanisms interact nonlinearly — they create new conflicts instead of resolving old ones.

**The diagnosis:** An 8-dimensional residual vector analysis shows the highest component is "cross-constraint consistency" at 1.83 — meaning the problem isn't any single constraint, it's the self-consistency of the ΛCDM repair strategy itself.

**Try it yourself (3 lines):**
```
pip install msce
msce check hubble --quick
```

The full analysis is in a Jupyter notebook (with Colab one-click):
[link to notebook]

I'm not a cosmologist — I build constraint-detection systems. If I've made errors in the physics constraints, please open an issue. The data and methodology are all open.

**Why this matters beyond Hubble:** Every field has theories that satisfy their authors' favorite constraints but fail others they didn't check. MSCE is an attempt to make "did you check ALL the constraints?" a one-line question.

— Xinhang

---

**Notes for posting:**
- Post at 00:00 UTC Monday (Sunday 8pm EST / Monday 8am Beijing)
- Reply to every comment in the first 6 hours
- Don't argue — if someone points out a physics error, thank them and fix it
- Link to the Colab notebook, not just the repo
