export type RequestStatus = 'success' | 'blocked_loop' | 'blocked_budget' | 'blocked_killswitch'
export type StatusFilter = 'all' | 'success' | 'blocked'
export type ProfileId = 'careful' | 'standard' | 'passive'
export type SimulationType = 'loop' | 'success' | 'budget'
export type ExportFormat = 'csv' | 'json'

export interface RequestRecord {
  id: number
  timestamp: string
  model: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd: number
  latency_ms: number
  prompt_hash: string
  status: RequestStatus | string
  blocked_reason: string | null
}

export interface Stats {
  total_requests: number
  success_requests: number
  blocked_requests: number
  blocked_budget_count: number
  blocked_loop_count: number
  blocked_killswitch_count: number
  total_spent: number
  total_prompt_tokens: number
  total_completion_tokens: number
  total_tokens: number
  avg_latency_ms: number
  all_time_spent: number
  all_time_tokens: number
  all_time_blocked_loops: number
  all_time_blocked_budget: number
  all_time_blocked_total: number
  saved_cost_estimate: number
  kill_switch_active: boolean
  active_profile: ProfileId | string
  hourly_limit: number
  daily_limit: number
  current_hourly_spend: number
  current_daily_spend: number
  hourly_budget_used_percent: number
  loop_threshold: number
  loop_window_size: number
  recent_hashes_count: number
  prices_count: number
}

export interface HistoryPoint {
  hour: string
  request_count: number
  success_count: number
  blocked_count: number
  spend_usd: number
  total_tokens: number
}

export type PriceTable = Record<string, { input: number; output: number }>

export interface LimitsUpdate {
  hourly_limit?: number
  daily_limit?: number
  loop_threshold?: number
  profile?: ProfileId
}

export interface LoopSimulationResult {
  simulation: 'loop'
  threshold: number
  results: { attempt: number; status: string }[]
}

export type SimulationResult =
  | LoopSimulationResult
  | { simulation: 'success'; model: string; cost: number }
  | { simulation: 'budget'; status: string }

export interface TokenGuardClient {
  getStats(): Promise<Stats>
  getRequests(status: StatusFilter): Promise<RequestRecord[]>
  getHistory(): Promise<HistoryPoint[]>
  getPrices(): Promise<PriceTable>
  toggleKillSwitch(): Promise<boolean>
  updateConfig(update: LimitsUpdate): Promise<void>
  simulate(type: SimulationType): Promise<SimulationResult>
  clearLogs(): Promise<void>
  exportLogs(format: ExportFormat, status: StatusFilter): Promise<void>
}
