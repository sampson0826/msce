export default function About() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="text-center mb-16">
        <h1 className="text-4xl sm:text-5xl font-bold mb-3">About StabilityBench</h1>
        <p className="text-dm-muted text-lg">LLM Recursive Stability — Who wins is one question. Who lasts is another.</p>
      </div>

      {/* What is Beta */}
      <section className="glass-card p-8 mb-8">
        <h2 className="text-2xl font-semibold mb-4">What is RSI?</h2>
        <div className="prose prose-invert max-w-none text-dm-muted space-y-4">
          <p>
            <span className="font-mono text-white font-semibold">RSI (Recursive Stability Index)</span> is
            a metric quantifying how fast an LLM's output quality degrades when it feeds on
            its own outputs across recursive generations.
          </p>
          <p>
            When an LLM's output becomes training data for the next generation,
            subtle quality deviations compound. This phenomenon, documented as
            <em>model collapse</em> in Nature (Shumailov et al., 2024), is measured
            by tracking constraint residual activity across generations.
          </p>
          <div className="bg-dm-surface/50 rounded-lg p-5 border border-dm-border/30">
            <div className="font-mono text-lg text-center text-white mb-2">
              RSI = constraint residual decay rate per generation
            </div>
            <div className="text-sm text-center">
              Lower RSI = slower decay = more stable model
            </div>
          </div>
        </div>
      </section>

      {/* The Framework */}
      <section className="glass-card p-8 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Constraint Residual Framework</h2>
        <div className="text-dm-muted space-y-4">
          <p>
            StabilityBench is built on the Constraint Residual Framework, which models LLM
            constraints as a 5-layer rule hierarchy (L-1 through L3). Each generation of
            recursive self-interaction causes some constraints to "drop" -- stop being
            respected by the model.
          </p>
          <p>
            The framework identifies constraint propagation paths, measures stability via
            the S_n score (per-capability stability), and derives the overall RSI
            from the cumulative constraint drop rate.
          </p>
          <h3 className="text-white font-medium mt-6">Key Concepts</h3>
          <ul className="list-disc list-inside space-y-2">
            <li><strong className="text-white">S_n</strong> -- Per-capability stability score at generation n (0-1 scale)</li>
            <li><strong className="text-white">RSI</strong> -- Recursive Stability Index: constraint drop rate per generation</li>
            <li><strong className="text-white">Generations</strong> -- Each recursive self-interaction cycle (Gen 0 through Gen 3)</li>
            <li><strong className="text-white">6 Capabilities</strong> -- Creative Writing, Math Reasoning, Code Generation, Logical Consistency, Factual Knowledge, General</li>
          </ul>
        </div>
      </section>

      {/* Methodology */}
      <section className="glass-card p-8 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Methodology</h2>
        <div className="text-dm-muted space-y-4">
          <p>
            Our Phase B evaluation tested n=100 models using a 6-model ranking approach across
            4 recursive generations. The evaluation pipeline:
          </p>
          <ol className="list-decimal list-inside space-y-2">
            <li>Extract active constraints from each model (256 initial constraints)</li>
            <li>Run recursive self-interaction for 4 generations</li>
            <li>Track constraint retention at each generation</li>
            <li>Measure per-capability S_n scores via LLM judge</li>
            <li>Compute RSI from cumulative constraint drop rate</li>
          </ol>
          <p className="mt-4">
            The results confirmed neural-level validation of the framework
            (lambda_C = 0.0415, R-squared = 0.884) and showed that creative writing
            is a universal bottleneck capability across all evaluated models.
          </p>
        </div>
      </section>

      {/* Scale Reference */}
      <section className="glass-card p-8 mb-8">
        <h2 className="text-2xl font-semibold mb-4">RSI Scale Reference</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-green-500/5 border border-green-500/20 rounded-lg p-5">
            <div className="font-mono text-green-400 font-bold mb-1">RSI &lt; 0.05</div>
            <div className="text-dm-muted text-sm">Excellent — Outputs can be safely used as training data. Minimal degradation across generations.</div>
          </div>
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-5">
            <div className="font-mono text-blue-400 font-bold mb-1">0.05 &lt;= RSI &lt; 0.10</div>
            <div className="text-dm-muted text-sm">Good — Manageable degradation. Active monitoring recommended for production training pipelines.</div>
          </div>
          <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-5">
            <div className="font-mono text-amber-400 font-bold mb-1">0.10 &lt;= RSI &lt; 0.15</div>
            <div className="text-dm-muted text-sm">Moderate — Noticeable quality loss. Requires data filtering before reuse in training.</div>
          </div>
          <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-5">
            <div className="font-mono text-red-400 font-bold mb-1">RSI &gt;= 0.15</div>
            <div className="text-dm-muted text-sm">Low — Rapid quality collapse. Outputs should NOT be used as training data without heavy curation.</div>
          </div>
        </div>
      </section>

      {/* Footer note */}
      <div className="text-center py-8 text-dm-muted text-sm">
        <p>StabilityBench is a research project based on the Constraint Residual Framework.</p>
        <p className="mt-1">For API access and detailed methodology, contact the research team.</p>
      </div>
    </div>
  )
}
