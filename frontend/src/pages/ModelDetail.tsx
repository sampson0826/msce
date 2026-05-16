import { useParams, Link } from 'react-router-dom'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { getModelBySlug, models } from '../data/models'
import { getRSIColor, getStatusColor, CAPABILITY_KEYS, CAPABILITY_LABELS } from '../data/types'

const StatusBadge = ({ status }: { status: string }) => {
  const styles: Record<string, string> = {
    healthy: 'bg-green-500/10 text-green-400 border-green-500/30',
    degrading: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    critical: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
    collapsed: 'bg-red-500/10 text-red-400 border-red-500/30',
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[status] || 'bg-dm-surface text-dm-muted'}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

export default function ModelDetail() {
  const { slug } = useParams<{ slug: string }>()
  const model = getModelBySlug(slug || '')

  if (!model) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <h1 className="text-4xl font-bold mb-4">Model Not Found</h1>
        <p className="text-dm-muted mb-8">No model with slug "{slug}" exists.</p>
        <Link to="/" className="btn-primary">Back to Leaderboard</Link>
      </div>
    )
  }

  // Prepare chart data for S_n trajectories
  const chartData = [0, 1, 2, 3].map((gen) => {
    const point: Record<string, number> = { generation: gen }
    CAPABILITY_KEYS.forEach((key) => {
      point[key] = model.capabilities[key].s_n[gen]
    })
    return point
  })

  // Constraint breakdown data
  const constraintData = model.generations.map((g) => ({
    generation: `Gen ${g.gen}`,
    remaining: g.constraints_remaining,
    dropped: g.cumulative_drop,
    total: g.total_constraints,
  }))

  const sortedByS3 = [...CAPABILITY_KEYS].sort(
    (a, b) => model.capabilities[a].s_n[3] - model.capabilities[b].s_n[3]
  )

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-dm-muted mb-8">
        <Link to="/" className="hover:text-dm-accent-light transition-colors">Leaderboard</Link>
        <span>/</span>
        <span className="text-white">{model.name}</span>
      </div>

      {/* Hero */}
      <div className="glass-card p-8 mb-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${
                model.family === 'DeepSeek' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
                model.family === 'OpenAI' ? 'bg-teal-500/10 text-teal-400 border-teal-500/30' :
                model.family === 'Llama' ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' :
                'bg-orange-500/10 text-orange-400 border-orange-500/30'
              }`}>
                {model.family}
              </span>
              <span className="text-dm-muted text-sm">Rank #{model.rank}</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold mb-2">{model.name}</h1>
          </div>
          <div className="text-center lg:text-right">
            <div className="text-dm-muted text-sm mb-1">Recursive Stability</div>
            <div className="font-mono text-5xl font-bold" style={{ color: getRSIColor(model.rsi) }}>
              {model.rsi.toFixed(4)}
            </div>
            <div className="text-dm-muted text-xs mt-1">RSI</div>
          </div>
        </div>
      </div>

      {/* S_n Trajectory Chart */}
      <div className="glass-card p-6 mb-8">
        <h2 className="text-lg font-semibold mb-1">Capability S_n Trajectories</h2>
        <p className="text-dm-muted text-sm mb-6">Per-capability stability score across 4 recursive generations</p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
              <XAxis
                dataKey="generation"
                stroke="#6b6b80"
                tick={{ fill: '#6b6b80', fontSize: 12 }}
                label={{ value: 'Generation', position: 'insideBottomRight', offset: -5, fill: '#6b6b80' }}
              />
              <YAxis
                stroke="#6b6b80"
                tick={{ fill: '#6b6b80', fontSize: 12 }}
                domain={[0.3, 1]}
                label={{ value: 'S_n', position: 'insideTopLeft', offset: -5, fill: '#6b6b80' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a2e',
                  border: '1px solid #2a2a3e',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Legend />
              {CAPABILITY_KEYS.map((key, i) => {
                const colors = ['#5e5ce6', '#40a9ff', '#30d158', '#faad14', '#ff7a45', '#ff4d4f']
                return (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    name={CAPABILITY_LABELS[key]}
                    stroke={colors[i]}
                    strokeWidth={2}
                    dot={{ r: 3, fill: colors[i] }}
                    activeDot={{ r: 5 }}
                  />
                )
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Capability Status Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {CAPABILITY_KEYS.map((key) => {
          const cap = model.capabilities[key]
          return (
            <div key={key} className="glass-card p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium">{CAPABILITY_LABELS[key]}</h3>
                <StatusBadge status={cap.status} />
              </div>
              <div className="flex items-end gap-1.5 mb-2">
                <span className="font-mono text-2xl font-bold" style={{ color: getStatusColor(cap.status) }}>
                  {cap.s_n[3].toFixed(2)}
                </span>
                <span className="text-dm-muted text-xs mb-0.5">S_3</span>
              </div>
              <div className="flex gap-1">
                {cap.s_n.map((s, i) => (
                  <div
                    key={i}
                    className="flex-1 h-1 rounded-full"
                    style={{
                      backgroundColor: getStatusColor(cap.status),
                      opacity: 0.3 + (i * 0.23),
                    }}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-1.5 text-xs text-dm-muted">
                <span>S₀={cap.s_n[0].toFixed(2)}</span>
                <span>S₃={cap.s_n[3].toFixed(2)}</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Constraint Breakdown */}
      <div className="glass-card p-6 mb-8">
        <h2 className="text-lg font-semibold mb-1">Per-Generation Constraint Breakdown</h2>
        <p className="text-dm-muted text-sm mb-6">Active constraints remaining after each recursive generation</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dm-border text-dm-muted text-xs uppercase tracking-wider">
                <th className="text-left py-3 px-4">Generation</th>
                <th className="text-right py-3 px-4">Constraints Remaining</th>
                <th className="text-right py-3 px-4">Cumulative Drop</th>
                <th className="text-right py-3 px-4">Retention</th>
                <th className="text-right py-3 px-4">Visual</th>
              </tr>
            </thead>
            <tbody>
              {constraintData.map((row) => {
                const retention = (row.remaining / row.total) * 100
                return (
                  <tr key={row.generation} className="border-b border-dm-border/30 hover:bg-dm-surface/20">
                    <td className="py-3 px-4 font-medium font-mono">{row.generation}</td>
                    <td className="py-3 px-4 text-right font-mono">{row.remaining} / {row.total}</td>
                    <td className="py-3 px-4 text-right font-mono text-red-400">-{row.dropped}</td>
                    <td className="py-3 px-4 text-right font-mono">{retention.toFixed(1)}%</td>
                    <td className="py-3 px-4">
                      <div className="w-full h-2 rounded-full bg-dm-surface">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${retention}%`,
                            backgroundColor: getRSIColor(model.rsi),
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Navigation to Compare */}
      <div className="text-center py-8">
        <Link to="/compare" className="btn-secondary">
          Compare with other models &rarr;
        </Link>
      </div>
    </div>
  )
}
