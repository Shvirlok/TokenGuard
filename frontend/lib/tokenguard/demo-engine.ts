import { DEMO_PRICES, PROFILES } from './constants'
import { isBlocked } from './format'
import type {
  ExportFormat,
  HistoryPoint,
  LimitsUpdate,
  RequestRecord,
  RequestStatus,
  SimulationResult,
  SimulationType,
  Stats,
  StatusFilter,
  TokenGuardClient,
} from './types'

const HOUR = 3_600_000
const LOOP_WINDOW_MS = 120_000
const TRAFFIC_MODELS = ['gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet', 'gemini-2.0-flash', 'deepseek-chat', 'qwen-plus']

const randInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min
const pick = <T,>(items: T[]) => items[Math.floor(Math.random() * items.length)]

function randomHash() {
  const bytes = new Uint8Array(32)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

class DemoEngine {
  private requests: RequestRecord[] = []
  private nextId = 1
  private hashes: { hash: string; t: number }[] = []
  private killSwitch = false
  private profile: string = 'standard'
  private hourlyLimit = 15
  private dailyLimit = 100
  private loopThreshold = 4

  constructor() {
    this.seed()
  }

  private cost(model: string, prompt: number, completion: number) {
    const price = DEMO_PRICES[model] ?? { input: 1, output: 3 }
    return (prompt * price.input + completion * price.output) / 1_000_000
  }

  private record(r: Omit<RequestRecord, 'id' | 'total_tokens' | 'timestamp'> & { t: number }) {
    const { t, ...rest } = r
    const rec: RequestRecord = {
      ...rest,
      id: this.nextId++,
      timestamp: new Date(t).toISOString(),
      total_tokens: rest.prompt_tokens + rest.completion_tokens,
      cost_usd: Number(rest.cost_usd.toFixed(6)),
    }
    this.requests.push(rec)
    return rec
  }

  private spendSince(since: number) {
    return this.requests.reduce((sum, r) => (Date.parse(r.timestamp) >= since ? sum + r.cost_usd : sum), 0)
  }

  private evaluate(hash: string, t: number): { status: RequestStatus; reason: string | null } {
    if (this.killSwitch) return { status: 'blocked_killswitch', reason: 'TokenGuard: kill switch active — all traffic halted' }
    if (this.spendSince(t - HOUR) >= this.hourlyLimit)
      return { status: 'blocked_budget', reason: 'TokenGuard: hourly budget limit exceeded' }
    if (this.loopThreshold > 0) {
      const repeats = this.hashes.filter((h) => h.hash === hash && t - h.t < LOOP_WINDOW_MS).length
      if (repeats >= this.loopThreshold - 1) {
        return {
          status: 'blocked_loop',
          reason: `TokenGuard: loop detected — ${repeats + 1}x identical prompt hash within rolling window`,
        }
      }
    }
    return { status: 'success', reason: null }
  }

  private process(model: string, prompt: number, completion: number, hash: string, latency: number, t = Date.now()) {
    const { status, reason } = this.evaluate(hash, t)
    const ok = status === 'success'
    if (ok) this.hashes.push({ hash, t })
    this.hashes = this.hashes.filter((h) => t - h.t < LOOP_WINDOW_MS * 5)
    return this.record({
      t,
      model,
      prompt_tokens: prompt,
      completion_tokens: ok ? completion : 0,
      cost_usd: ok ? this.cost(model, prompt, completion) : 0,
      latency_ms: ok ? latency : 0,
      prompt_hash: hash,
      status,
      blocked_reason: reason,
    })
  }

  private seed() {
    const now = Date.now()
    for (let h = 23; h >= 0; h--) {
      const start = now - (h + 1) * HOUR
      const busy = h >= 6 && h <= 14
      const count = h === 0 ? 3 : randInt(busy ? 3 : 1, busy ? 8 : 4)
      const times = Array.from({ length: count }, () => start + Math.random() * HOUR).sort((a, b) => a - b)
      for (const t of times) {
        this.process(pick(TRAFFIC_MODELS), randInt(60, 1800), randInt(80, 900), randomHash(), randInt(140, 720), t)
      }
      if (h === 17 || h === 5) {
        const hash = randomHash()
        const base = start + HOUR * 0.6
        for (let i = 0; i < this.loopThreshold; i++) this.process('gpt-4o', 92, 150, hash, 180 + i * 20, base + i * 4000)
      }
    }
    this.requests.sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp))
    this.requests.forEach((r, i) => (r.id = i + 1))
    this.nextId = this.requests.length + 1
    this.hashes = []
  }

  tick() {
    if (Math.random() > 0.4) return
    this.process(pick(TRAFFIC_MODELS), randInt(60, 1600), randInt(80, 800), randomHash(), randInt(140, 640))
  }

  stats(): Stats {
    const now = Date.now()
    const hour = this.requests.filter((r) => Date.parse(r.timestamp) >= now - HOUR)
    const count = (list: RequestRecord[], status: string) => list.filter((r) => r.status === status).length
    const ok = hour.filter((r) => r.status === 'success')
    const hourly = this.spendSince(now - HOUR)
    const saved = this.requests
      .filter((r) => r.status === 'blocked_loop' || r.status === 'blocked_budget')
      .reduce((s, r) => s + this.cost(r.model, r.prompt_tokens, 150), 0)

    return {
      total_requests: hour.length,
      success_requests: ok.length,
      blocked_requests: hour.length - ok.length,
      blocked_budget_count: count(hour, 'blocked_budget'),
      blocked_loop_count: count(hour, 'blocked_loop'),
      blocked_killswitch_count: count(hour, 'blocked_killswitch'),
      total_spent: Number(hourly.toFixed(6)),
      total_prompt_tokens: hour.reduce((s, r) => s + r.prompt_tokens, 0),
      total_completion_tokens: hour.reduce((s, r) => s + r.completion_tokens, 0),
      total_tokens: hour.reduce((s, r) => s + r.total_tokens, 0),
      avg_latency_ms: ok.length ? Math.round((ok.reduce((s, r) => s + r.latency_ms, 0) / ok.length) * 10) / 10 : 0,
      all_time_spent: Number(this.requests.reduce((s, r) => s + r.cost_usd, 0).toFixed(6)),
      all_time_tokens: this.requests.reduce((s, r) => s + r.total_tokens, 0),
      all_time_blocked_loops: count(this.requests, 'blocked_loop'),
      all_time_blocked_budget: count(this.requests, 'blocked_budget'),
      all_time_blocked_total: this.requests.filter((r) => isBlocked(r.status)).length,
      saved_cost_estimate: Number(saved.toFixed(6)),
      kill_switch_active: this.killSwitch,
      active_profile: this.profile,
      hourly_limit: this.hourlyLimit,
      daily_limit: this.dailyLimit,
      current_hourly_spend: Number(hourly.toFixed(6)),
      current_daily_spend: Number(this.spendSince(now - 24 * HOUR).toFixed(6)),
      hourly_budget_used_percent: this.hourlyLimit > 0 ? Math.round((hourly / this.hourlyLimit) * 10000) / 100 : 0,
      loop_threshold: this.loopThreshold,
      loop_window_size: 10,
      recent_hashes_count: this.hashes.length,
      prices_count: Object.keys(DEMO_PRICES).length,
    }
  }

  list(status: StatusFilter) {
    const filtered = this.requests.filter((r) =>
      status === 'all' ? true : status === 'success' ? r.status === 'success' : isBlocked(r.status),
    )
    return filtered.slice(-100).reverse()
  }

  history(): HistoryPoint[] {
    const end = new Date()
    end.setMinutes(0, 0, 0)
    const points: HistoryPoint[] = []
    for (let i = 23; i >= 0; i--) {
      const start = end.getTime() - i * HOUR
      const bucket = this.requests.filter((r) => {
        const t = Date.parse(r.timestamp)
        return t >= start && t < start + HOUR
      })
      points.push({
        hour: new Date(start).toISOString(),
        request_count: bucket.length,
        success_count: bucket.filter((r) => r.status === 'success').length,
        blocked_count: bucket.filter((r) => isBlocked(r.status)).length,
        spend_usd: Number(bucket.reduce((s, r) => s + r.cost_usd, 0).toFixed(6)),
        total_tokens: bucket.reduce((s, r) => s + r.total_tokens, 0),
      })
    }
    return points
  }

  toggleKill() {
    this.killSwitch = !this.killSwitch
    return this.killSwitch
  }

  configure(update: LimitsUpdate) {
    if (update.profile) {
      const p = PROFILES.find((x) => x.id === update.profile)
      if (p) {
        this.profile = p.id
        this.hourlyLimit = p.hourly_limit
        this.dailyLimit = p.daily_limit
        this.loopThreshold = p.loop_threshold
      }
    }
    if (update.hourly_limit !== undefined) this.hourlyLimit = update.hourly_limit
    if (update.daily_limit !== undefined) this.dailyLimit = update.daily_limit
    if (update.loop_threshold !== undefined) this.loopThreshold = update.loop_threshold
  }

  simulate(type: SimulationType): SimulationResult {
    if (type === 'loop') {
      const hash = randomHash()
      const attempts = Math.max(this.loopThreshold, 3)
      const results = Array.from({ length: attempts }, (_, i) => {
        const rec = this.process('gpt-4o', 92, 150, hash, 160 + (i + 1) * 20, Date.now() + i)
        return { attempt: i + 1, status: rec.status === 'success' ? 'allowed' : rec.status }
      })
      return { simulation: 'loop', threshold: this.loopThreshold, results }
    }
    if (type === 'success') {
      const rec = this.process('gpt-4o', 85, 140, randomHash(), 210)
      return { simulation: 'success', model: rec.model, cost: rec.cost_usd }
    }
    this.record({
      t: Date.now(),
      model: 'gpt-4o',
      prompt_tokens: 30000,
      completion_tokens: 0,
      cost_usd: 0,
      latency_ms: 0,
      prompt_hash: randomHash(),
      status: 'blocked_budget',
      blocked_reason: 'TokenGuard: budget limit exceeded',
    })
    return { simulation: 'budget', status: 'blocked_budget' }
  }

  clear() {
    this.requests = []
    this.hashes = []
    this.nextId = 1
  }

  export(format: ExportFormat, status: StatusFilter) {
    const rows = this.list(status)
    if (format === 'json') return JSON.stringify({ requests: rows, count: rows.length }, null, 2)
    const fields = [
      'id',
      'timestamp',
      'model',
      'prompt_tokens',
      'completion_tokens',
      'total_tokens',
      'cost_usd',
      'latency_ms',
      'prompt_hash',
      'status',
      'blocked_reason',
    ] as const
    const escape = (v: unknown) => {
      const s = String(v ?? '')
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
    }
    return [fields.join(','), ...rows.map((r) => fields.map((f) => escape(r[f])).join(','))].join('\n')
  }
}

let engine: DemoEngine | null = null
export function getDemoEngine() {
  engine ??= new DemoEngine()
  return engine
}

function download(content: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function createDemoClient(): TokenGuardClient {
  const e = getDemoEngine()
  return {
    getStats: async () => e.stats(),
    getRequests: async (status) => e.list(status),
    getHistory: async () => e.history(),
    getPrices: async () => DEMO_PRICES,
    toggleKillSwitch: async () => e.toggleKill(),
    updateConfig: async (u) => e.configure(u),
    simulate: async (t) => e.simulate(t),
    clearLogs: async () => e.clear(),
    exportLogs: async (format, status) =>
      download(
        e.export(format, status),
        `tokenguard_requests.${format}`,
        format === 'csv' ? 'text/csv' : 'application/json',
      ),
  }
}
