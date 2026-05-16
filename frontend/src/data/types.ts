export interface ModelEntry {
  name: string
  family: string
  rsi: number
  rank: number
  trend: 'up' | 'down' | 'stable'
  slug: string
  capabilities: CapabilityData
  generations: GenerationData[]
}

export interface CapabilityData {
  [key: string]: CapabilityStatus
}

export interface CapabilityStatus {
  label: string
  status: 'healthy' | 'degrading' | 'critical' | 'collapsed'
  s_n: number[]
}

export interface GenerationData {
  gen: number
  constraints_remaining: number
  total_constraints: number
  cumulative_drop: number
}

export const CAPABILITY_KEYS = [
  'creative_writing',
  'math_reasoning',
  'code_generation',
  'logical_consistency',
  'factual_knowledge',
  'general',
] as const

export const CAPABILITY_LABELS: Record<string, string> = {
  creative_writing: 'Creative Writing',
  math_reasoning: 'Math Reasoning',
  code_generation: 'Code Generation',
  logical_consistency: 'Logical Consistency',
  factual_knowledge: 'Factual Knowledge',
  general: 'General',
}

export function getRSIColor(rsi: number): string {
  if (rsi <= 0.05) return '#30d158'
  if (rsi <= 0.10) return '#40a9ff'
  if (rsi <= 0.15) return '#faad14'
  return '#ff4d4f'
}

export function getRSIBgClass(rsi: number): string {
  if (rsi <= 0.05) return 'bg-green-500/10 border-green-500/30 text-green-400'
  if (rsi <= 0.10) return 'bg-blue-500/10 border-blue-500/30 text-blue-400'
  if (rsi <= 0.15) return 'bg-amber-500/10 border-amber-500/30 text-amber-400'
  return 'bg-red-500/10 border-red-500/30 text-red-400'
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'healthy': return '#30d158'
    case 'degrading': return '#faad14'
    case 'critical': return '#ff7a45'
    case 'collapsed': return '#ff4d4f'
    default: return '#6b6b80'
  }
}

export function getRSIRowBg(rsi: number): string {
  if (rsi <= 0.05) return 'border-l-green-500'
  if (rsi <= 0.10) return 'border-l-blue-500'
  if (rsi <= 0.15) return 'border-l-amber-500'
  return 'border-l-red-500'
}
