import { useState } from 'react'
import { Link } from 'react-router-dom'
import { models } from '../data/models'
import { getRSIColor, getStatusColor, CAPABILITY_KEYS, CAPABILITY_LABELS } from '../data/types'

export default function Compare() {
  const [selected, setSelected] = useState<string[]>([])

  const toggleModel = (slug: string) => {
    setSelected((prev) => {
      if (prev.includes(slug)) return prev.filter((s) => s !== slug)
      if (prev.length >= 4) return prev
      return [...prev, slug]
    })
  }

  const selectedModels = models.filter((m) => selected.includes(m.slug))

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Hero */}
      <div className="text-center mb-12">
        <h1 className="text-4xl sm:text-5xl font-bold mb-3">Model Comparison</h1>
        <p className="text-dm-muted max-w-xl mx-auto">
          Select 2-4 models to compare stability metrics, capability degradation, and constraint drops side by side.
        </p>
      </div>

      {/* Model Selection */}
      <div className="glass-card p-6 mb-10">
        <h2 className="text-lg font-semibold mb-4">
          Select Models
          <span className="text-dm-muted text-sm font-normal ml-2">({selected.length}/4 selected)</span>
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {models.map((model) => {
            const isSelected = selected.includes(model.slug)
            return (
              <button
                key={model.slug}
                onClick={() => toggleModel(model.slug)}
                className={`p-3 rounded-lg border text-left transition-all duration-200 ${
                  isSelected
                    ? 'border-dm-accent bg-dm-accent/10 text-white'
                    : 'border-dm-border bg-dm-surface/30 text-dm-muted hover:border-dm-border hover:text-white'
                }`}
              >
                <div className="text-xs font-medium truncate">{model.name}</div>
                <div className="text-xs mt-1 font-mono" style={{ color: getRSIColor(model.rsi) }}>
                  RSI={model.rsi.toFixed(4)}
                </div>
                {isSelected && (
                  <div className="mt-1 text-xs text-dm-accent-light">Selected</div>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {selectedModels.length === 0 && (
        <div className="text-center py-16 glass-card">
          <div className="text-dm-muted text-5xl mb-4">&#8625;</div>
          <p className="text-dm-muted text-lg">Select models above to compare</p>
          <p className="text-dm-muted text-sm mt-1">Choose 2-4 models to begin comparison</p>
        </div>
      )}

      {selectedModels.length > 0 && (
        <>
          {/* Beta Comparison */}
          <div className="glass-card p-6 mb-8">
            <h2 className="text-lg font-semibold mb-6">RSI Comparison</h2>
            <div className="space-y-4">
              {selectedModels
                .sort((a, b) => a.rsi - b.rsi)
                .map((model, idx) => {
                  const maxRsi = Math.max(...selectedModels.map((m) => m.rsi))
                  const barWidth = (model.rsi / maxRsi) * 100
                  return (
                    <div key={model.slug} className="flex items-center gap-4">
                      <div className="w-6 text-right font-mono text-sm text-dm-muted">
                        {idx + 1}
                      </div>
                      <div className="w-40 text-sm font-medium truncate">
                        <Link to={`/model/${model.slug}`} className="hover:text-dm-accent-light transition-colors">
                          {model.name}
                        </Link>
                      </div>
                      <div className="flex-1 h-8 rounded-md bg-dm-surface relative overflow-hidden">
                        <div
                          className="h-full rounded-md transition-all duration-700 flex items-center justify-end pr-2"
                          style={{
                            width: `${Math.max(barWidth, 8)}%`,
                            backgroundColor: getRSIColor(model.rsi),
                            opacity: 0.15 + 0.6 * (model.rsi / maxRsi),
                          }}
                        />
                      </div>
                      <div className="w-24 text-right font-mono text-sm font-semibold" style={{ color: getRSIColor(model.rsi) }}>
                        {model.rsi.toFixed(4)}
                      </div>
                    </div>
                  )
                })}
            </div>
          </div>

          {/* Per-Capability S_3 Heatmap */}
          <div className="glass-card p-6 mb-8">
            <h2 className="text-lg font-semibold mb-6">Per-Capability S_3 Heatmap</h2>
            <p className="text-dm-muted text-sm mb-4">
              Final generation S_3 scores across capabilities. Greener = healthier.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-dm-border">
                    <th className="text-left py-3 px-4 text-dm-muted text-xs uppercase tracking-wider">Capability</th>
                    {selectedModels.map((m) => (
                      <th key={m.slug} className="text-right py-3 px-4 text-dm-muted text-xs uppercase tracking-wider">
                        {m.name.length > 16 ? m.name.slice(0, 14) + '..' : m.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CAPABILITY_KEYS.map((key) => (
                    <tr key={key} className="border-b border-dm-border/30 hover:bg-dm-surface/20">
                      <td className="py-3 px-4 font-medium">{CAPABILITY_LABELS[key]}</td>
                      {selectedModels.map((m) => {
                        const cap = m.capabilities[key]
                        const s3 = cap.s_n[3]
                        const bgOpacity = s3 < 0.5 ? 0.4 : s3 < 0.7 ? 0.25 : 0.12
                        const color = getStatusColor(cap.status)
                        const isWorst = Math.min(...selectedModels.map((sm) => sm.capabilities[key].s_n[3])) === s3
                        const isBest = Math.max(...selectedModels.map((sm) => sm.capabilities[key].s_n[3])) === s3
                        return (
                          <td
                            key={m.slug}
                            className="py-3 px-4 text-right font-mono text-sm relative"
                            style={{
                              backgroundColor: `${color}${Math.round(bgOpacity * 255).toString(16).padStart(2, '0')}`,
                            }}
                          >
                            <span className="font-semibold" style={{ color }}>
                              {s3.toFixed(2)}
                            </span>
                            {isBest && selectedModels.length > 1 && (
                              <span className="ml-1 text-green-400 text-xs" title="Best in class">&#9650;</span>
                            )}
                            {isWorst && selectedModels.length > 1 && (
                              <span className="ml-1 text-red-400 text-xs" title="Worst in class">&#9660;</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Side-by-side constraint drop summary */}
          <div className="glass-card p-6 mb-8">
            <h2 className="text-lg font-semibold mb-6">Generation-3 Constraint Summary</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {selectedModels.map((m) => {
                const g3 = m.generations[3]
                const retention = ((g3.constraints_remaining / g3.total_constraints) * 100).toFixed(1)
                return (
                  <div key={m.slug} className="bg-dm-surface/50 rounded-lg p-4 border border-dm-border/30">
                    <div className="text-sm font-medium mb-2 truncate">{m.name}</div>
                    <div className="font-mono text-xl font-bold" style={{ color: getRSIColor(m.rsi) }}>
                      {retention}%
                    </div>
                    <div className="text-dm-muted text-xs mt-2">
                      {g3.constraints_remaining} / {g3.total_constraints} constraints retained
                    </div>
                    <div className="text-red-400 text-xs">
                      -{g3.cumulative_drop} cumulative drop
                    </div>
                    <div className="mt-3 w-full h-1.5 rounded-full bg-dm-bg">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${retention}%`,
                          backgroundColor: getRSIColor(m.rsi),
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
