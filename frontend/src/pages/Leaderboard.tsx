import { Link } from 'react-router-dom'
import { models, families } from '../data/models'
import { getRSIColor, getRSIBgClass, getRSIRowBg } from '../data/types'

const TrendIcon = ({ trend }: { trend: 'up' | 'down' | 'stable' }) => {
  if (trend === 'up') {
    return (
      <span className="text-red-400" title="Getting worse (RSI increasing)">
        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      </span>
    )
  }
  if (trend === 'down') {
    return (
      <span className="text-green-400" title="Improving (RSI decreasing)">
        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6" />
        </svg>
      </span>
    )
  }
  return (
    <span className="text-dm-muted" title="Stable">
      <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
      </svg>
    </span>
  )
}

const FamilyBadge = ({ family }: { family: string }) => {
  const colors: Record<string, string> = {
    DeepSeek: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    OpenAI: 'bg-teal-500/10 text-teal-400 border-teal-500/30',
    Llama: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    Claude: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
  }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${colors[family] || 'bg-dm-surface text-dm-muted border-dm-border'}`}>
      {family}
    </span>
  )
}

export default function Leaderboard() {
  return (
    <div>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-dm-accent/5 via-transparent to-transparent pointer-events-none" />
        <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full bg-dm-accent/3 blur-3xl pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 relative z-10">
          <div className="text-center">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-dm-accent/10 border border-dm-accent/20 mb-8">
              <span className="w-2 h-2 rounded-full bg-dm-accent-light animate-pulse" />
              <span className="text-dm-accent-light text-sm font-medium">Phase B Results Live</span>
            </div>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight mb-4">
              Stability<span className="text-dm-accent">Bench</span>
            </h1>
            <p className="text-xl sm:text-2xl text-dm-muted max-w-2xl mx-auto mb-3">
              LLM Recursive Stability <span className="text-white font-mono">RSI</span> Leaderboard
            </p>
            <p className="text-sm text-dm-muted max-w-xl mx-auto">
              Tracking recursive stability across frontier language models under recursive self-interaction.
              Lower RSI = more stable across generations.
            </p>
          </div>

          {/* Stats bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-14 max-w-3xl mx-auto">
            <div className="glass-card p-4 text-center">
              <div className="stat-value text-dm-accent-light">{models.length}</div>
              <div className="stat-label mt-1">Models Tracked</div>
            </div>
            <div className="glass-card p-4 text-center">
              <div className="stat-value text-green-400">{families.length}</div>
              <div className="stat-label mt-1">Model Families</div>
            </div>
            <div className="glass-card p-4 text-center">
              <div className="stat-value text-blue-400">{models.filter((m) => m.rsi <= 0.05).length}</div>
              <div className="stat-label mt-1">RSI &lt; 0.05</div>
            </div>
            <div className="glass-card p-4 text-center">
              <div className="stat-value text-amber-400">0.088</div>
              <div className="stat-label mt-1">Median RSI</div>
            </div>
          </div>
        </div>
      </section>

      {/* Leaderboard Table */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">Stability Ranking</h2>
          <div className="flex items-center gap-4 text-xs text-dm-muted">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
              Stable
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
              Mild
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
              Moderate
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
              High Decay
            </span>
          </div>
        </div>

        <div className="glass-card overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-12 gap-4 px-6 py-3 bg-dm-surface/50 border-b border-dm-border text-xs text-dm-muted uppercase tracking-wider font-medium">
            <div className="col-span-1">Rank</div>
            <div className="col-span-5">Model</div>
            <div className="col-span-2">Family</div>
            <div className="col-span-2 text-right">RSI</div>
            <div className="col-span-1 text-center">Trend</div>
            <div className="col-span-1 text-right">Detail</div>
          </div>

          {/* Table rows */}
          {models.map((model) => (
            <div
              key={model.slug}
              className={`grid grid-cols-12 gap-4 px-6 py-4 border-b border-dm-border/50 border-l-4 ${getRSIRowBg(model.rsi)} hover:bg-dm-surface/30 transition-colors items-center`}
            >
              {/* Rank */}
              <div className="col-span-1">
                <span className={`font-mono text-lg font-bold ${
                  model.rank === 1 ? 'text-amber-400' :
                  model.rank === 2 ? 'text-dm-muted' :
                  model.rank === 3 ? 'text-amber-700' :
                  'text-dm-muted'
                }`}>
                  {model.rank === 1 ? '🥇' : model.rank === 2 ? '🥈' : model.rank === 3 ? '🥉' : `#${model.rank}`}
                </span>
              </div>

              {/* Model name */}
              <div className="col-span-5">
                <Link
                  to={`/model/${model.slug}`}
                  className="text-white font-medium hover:text-dm-accent-light transition-colors"
                >
                  {model.name}
                </Link>
              </div>

              {/* Family badge */}
              <div className="col-span-2">
                <FamilyBadge family={model.family} />
              </div>

              {/* Beta value */}
              <div className="col-span-2 text-right">
                <span className={`font-mono text-sm font-semibold`} style={{ color: getRSIColor(model.rsi) }}>
                  {model.rsi.toFixed(4)}
                </span>
              </div>

              {/* Trend */}
              <div className="col-span-1 flex justify-center">
                <TrendIcon trend={model.trend} />
              </div>

              {/* Detail link */}
              <div className="col-span-1 text-right">
                <Link
                  to={`/model/${model.slug}`}
                  className="text-dm-muted hover:text-dm-accent-light transition-colors text-sm"
                >
                  View &rarr;
                </Link>
              </div>
            </div>
          ))}
        </div>

        {/* Beta scale visualization */}
        <div className="mt-8 glass-card p-6">
          <h3 className="text-sm font-medium text-dm-muted mb-4">RSI Scale Interpretation</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="w-20 text-xs text-green-400 font-mono">0.00-0.05</span>
              <div className="flex-1 h-2 rounded-full bg-green-500/20">
                <div className="h-full w-[5%] rounded-full bg-green-500" />
              </div>
              <span className="text-xs text-green-400">Stable</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-20 text-xs text-blue-400 font-mono">0.05-0.10</span>
              <div className="flex-1 h-2 rounded-full bg-blue-500/20">
                <div className="h-full w-[45%] rounded-full bg-blue-500" />
              </div>
              <span className="text-xs text-blue-400">Mild Decay</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-20 text-xs text-amber-400 font-mono">0.10-0.15</span>
              <div className="flex-1 h-2 rounded-full bg-amber-500/20">
                <div className="h-full w-[85%] rounded-full bg-amber-500" />
              </div>
              <span className="text-xs text-amber-400">Moderate Decay</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-20 text-xs text-red-400 font-mono">0.15+</span>
              <div className="flex-1 h-2 rounded-full bg-red-500/20">
                <div className="h-full w-full rounded-full bg-red-500" />
              </div>
              <span className="text-xs text-red-400">Severe Decay</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
