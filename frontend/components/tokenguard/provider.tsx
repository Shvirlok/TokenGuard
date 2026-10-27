'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import useSWR, { useSWRConfig } from 'swr'
import { toast } from 'sonner'
import { createDemoClient, getDemoEngine } from '@/lib/tokenguard/demo-engine'
import { isBlocked } from '@/lib/tokenguard/format'
import { createLiveClient, detectProxy } from '@/lib/tokenguard/live-client'
import type {
  ExportFormat,
  HistoryPoint,
  LimitsUpdate,
  PriceTable,
  ProfileId,
  RequestRecord,
  SimulationType,
  Stats,
  StatusFilter,
} from '@/lib/tokenguard/types'
import { PROFILES } from '@/lib/tokenguard/constants'

export type ConnectionMode = 'connecting' | 'live' | 'demo'

interface TokenGuardContextValue {
  mode: ConnectionMode
  baseUrl: string
  stats?: Stats
  requests?: RequestRecord[]
  history?: HistoryPoint[]
  prices?: PriceTable
  polling: boolean
  setPolling: (value: boolean) => void
  statusFilter: StatusFilter
  setStatusFilter: (value: StatusFilter) => void
  soundEnabled: boolean
  setSoundEnabled: (value: boolean) => void
  pending: string | null
  retryConnection: () => void
  toggleKillSwitch: () => Promise<void>
  setProfile: (profile: ProfileId) => Promise<void>
  updateLimits: (update: LimitsUpdate) => Promise<boolean>
  simulate: (type: SimulationType) => Promise<void>
  clearLogs: () => Promise<void>
  exportLogs: (format: ExportFormat) => Promise<void>
}

const TokenGuardContext = createContext<TokenGuardContextValue | null>(null)

export function useTokenGuard() {
  const ctx = useContext(TokenGuardContext)
  if (!ctx) throw new Error('useTokenGuard must be used inside TokenGuardProvider')
  return ctx
}

function playAlert() {
  try {
    const ctx = new AudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'triangle'
    osc.frequency.setValueAtTime(880, ctx.currentTime)
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.25)
    gain.gain.setValueAtTime(0.12, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3)
    osc.connect(gain).connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + 0.3)
  } catch {}
}

export function TokenGuardProvider({ children }: { children: React.ReactNode }) {
  const { mutate } = useSWRConfig()
  const [polling, setPolling] = useState(true)
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [soundEnabled, setSoundEnabled] = useState(false)
  const [pending, setPending] = useState<string | null>(null)

  const connection = useSWR('tg:connection', detectProxy, {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
  })
  const mode: ConnectionMode = connection.isLoading ? 'connecting' : typeof connection.data === 'string' ? 'live' : 'demo'
  const baseUrl = typeof connection.data === 'string' ? connection.data : ''

  const client = useMemo(() => {
    if (mode === 'live') return createLiveClient(baseUrl)
    if (mode === 'demo') return createDemoClient()
    return null
  }, [mode, baseUrl])

  const interval = mode === 'demo' ? (polling ? 2000 : 0) : 0
  const key = useCallback(
    (name: string, ...extra: string[]) => (client ? ['tg', mode, baseUrl, name, ...extra] : null),
    [client, mode, baseUrl],
  )

  const { data: stats } = useSWR(key('stats'), () => client!.getStats(), { refreshInterval: interval })
  const { data: requests } = useSWR(key('requests', statusFilter), () => client!.getRequests(statusFilter), {
    refreshInterval: interval,
    keepPreviousData: true,
  })
  const { data: history } = useSWR(key('history'), () => client!.getHistory(), {
    refreshInterval: polling ? 10000 : 0,
  })
  const { data: prices } = useSWR(key('prices'), () => client!.getPrices(), { revalidateOnFocus: false })

  const refreshAll = useCallback(
    () => mutate((k) => Array.isArray(k) && k[0] === 'tg' && k[3] !== 'prices'),
    [mutate],
  )

  useEffect(() => {
    if (mode !== 'demo' || !polling) return
    const id = setInterval(() => {
      getDemoEngine().tick()
      refreshAll()
    }, 4000)
    return () => clearInterval(id)
  }, [mode, polling, refreshAll])

  useEffect(() => {
    if (mode !== 'live' || !baseUrl || !polling) return
    if (typeof window === 'undefined' || !('EventSource' in window)) return

    let es: EventSource | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let retryDelay = 1000

    const connectSSE = () => {
      try {
        es = new EventSource(`${baseUrl}/api/events`)

        es.onopen = () => {
          retryDelay = 1000
        }

        es.addEventListener('stats', (e) => {
          try {
            const data = JSON.parse(e.data)
            mutate(key('stats'), (prev: any) => ({ ...prev, ...data }), false)
          } catch {}
        })

        es.addEventListener('request_logged', (e) => {
          try {
            const newReq: RequestRecord = JSON.parse(e.data)

            const shouldInclude =
              statusFilter === 'all' ||
              (statusFilter === 'success' && newReq.status === 'success') ||
              (statusFilter === 'blocked' && isBlocked(newReq.status))

            if (shouldInclude) {
              mutate(
                key('requests', statusFilter),
                (prev: RequestRecord[] = []) => [newReq, ...prev.filter((r) => r.id !== newReq.id)],
                false,
              )
            }

            if (statusFilter !== 'all') {
              mutate(
                key('requests', 'all'),
                (prev: RequestRecord[] = []) => [newReq, ...prev.filter((r) => r.id !== newReq.id)],
                false,
              )
            }

            mutate(
              key('stats'),
              (prev: Stats | undefined) => {
                if (!prev) return prev
                const isSuccess = newReq.status === 'success'
                const isBlockedReq = !isSuccess
                const isLoop =
                  newReq.status === 'blocked_loop' ||
                  newReq.status === 'blocked_tool_loop' ||
                  (newReq.blocked_reason?.toLowerCase().includes('loop') ?? false)
                const isBudget = newReq.status === 'blocked_budget'

                const avoidedTokens = isLoop ? Math.max((newReq.prompt_tokens || 110) * 3 + 1200, 1500) : 0
                const avoidedCost = isLoop ? Math.max(avoidedTokens * 0.000005, 0.0075) : (isBudget ? 0.015 : 0)

                return {
                  ...prev,
                  total_requests: prev.total_requests + 1,
                  success_requests: isSuccess ? prev.success_requests + 1 : prev.success_requests,
                  blocked_requests: isBlockedReq ? prev.blocked_requests + 1 : prev.blocked_requests,
                  blocked_loop_count: isLoop ? (prev.blocked_loop_count || 0) + 1 : prev.blocked_loop_count,
                  all_time_blocked_loops: isLoop ? (prev.all_time_blocked_loops || 0) + 1 : prev.all_time_blocked_loops,
                  all_time_blocked_total: isBlockedReq ? (prev.all_time_blocked_total || 0) + 1 : prev.all_time_blocked_total,
                  saved_cost_estimate: (prev.saved_cost_estimate || 0) + avoidedCost,
                  total_spent: isSuccess ? prev.total_spent + (newReq.cost_usd || 0) : prev.total_spent,
                  total_tokens: isSuccess ? prev.total_tokens + (newReq.total_tokens || 0) : prev.total_tokens,
                }
              },
              false,
            )

            if (soundEnabled && isBlocked(newReq.status)) {
              playAlert()
            }
          } catch {}
        })

        es.addEventListener('state_change', (e) => {
          try {
            const data = JSON.parse(e.data)
            mutate(key('stats'), (prev: any) => ({ ...prev, ...data }), false)
          } catch {}
        })

        es.addEventListener('logs_cleared', () => {
          mutate(key('requests', statusFilter), [], false)
          mutate(key('requests', 'all'), [], false)
          refreshAll()
        })

        es.onerror = () => {
          es?.close()
          es = null
          reconnectTimer = setTimeout(() => {
            retryDelay = Math.min(retryDelay * 1.5, 10000)
            connectSSE()
          }, retryDelay)
        }
      } catch {
        reconnectTimer = setTimeout(connectSSE, 3000)
      }
    }

    connectSSE()

    return () => {
      es?.close()
      if (reconnectTimer) clearTimeout(reconnectTimer)
    }
  }, [mode, baseUrl, polling, statusFilter, soundEnabled, key, mutate, refreshAll])

  const lastSeenId = useRef<number | null>(null)
  useEffect(() => {
    if (!requests?.length) return
    const maxId = Math.max(...requests.map((r) => r.id))
    const previous = lastSeenId.current
    lastSeenId.current = maxId
    if (previous === null || !soundEnabled) return
    if (requests.some((r) => r.id > previous && isBlocked(r.status))) playAlert()
  }, [requests, soundEnabled])

  const run = useCallback(
    async <T,>(label: string, fn: () => Promise<T>): Promise<T | undefined> => {
      setPending(label)
      try {
        const result = await fn()
        await refreshAll()
        return result
      } catch (err) {
        toast.error('Request failed', { description: err instanceof Error ? err.message : String(err) })
        return undefined
      } finally {
        setPending(null)
      }
    },
    [refreshAll],
  )

  const value: TokenGuardContextValue = {
    mode,
    baseUrl,
    stats,
    requests,
    history,
    prices,
    polling,
    setPolling,
    statusFilter,
    setStatusFilter,
    soundEnabled,
    setSoundEnabled,
    pending,
    retryConnection: () => connection.mutate(),
    toggleKillSwitch: async () => {
      if (!client) return
      const active = await run('kill', () => client.toggleKillSwitch())
      if (active === undefined) return
      if (active) toast.error('Kill switch engaged', { description: 'All proxy traffic is now blocked.' })
      else toast.success('Traffic resumed', { description: 'Requests are flowing through TokenGuard again.' })
    },
    setProfile: async (profile) => {
      if (!client) return
      const ok = await run(`profile:${profile}`, async () => {
        await client.updateConfig({ profile })
        return true
      })
      const def = PROFILES.find((p) => p.id === profile)
      if (ok && def) toast.success(`${def.name} profile active`, { description: def.description })
    },
    updateLimits: async (update) => {
      if (!client) return false
      const ok = await run('config', async () => {
        await client.updateConfig(update)
        return true
      })
      if (ok) toast.success('Guard limits updated')
      return Boolean(ok)
    },
    simulate: async (type) => {
      if (!client) return
      const result = await run(`sim:${type}`, () => client.simulate(type))
      if (!result) return
      if (result.simulation === 'tool_loop') {
        const toolName = result.tool_name || 'read_file'
        toast.error('Tool Ping-Pong Intercepted', {
          description: `[TOOL RETRY LOOP: HARD STOP] Intercepted repeated ${toolName} loop — circuit breaker tripped!`,
        })
      } else if (result.simulation === 'loop') {
        const blockedAt = result.results.find((r) => r.status !== 'allowed')
        if (blockedAt)
          toast.warning('Runaway loop intercepted', {
            description: `Identical call #${blockedAt.attempt} was blocked (${blockedAt.status.replace('blocked_', '')}).`,
          })
        else
          toast.info('Loop tracked, not blocked', {
            description: `${result.results.length} identical calls logged — loop blocking is off for this profile.`,
          })
      } else if (result.simulation === 'success') {
        toast.success('Simulated request passed', { description: `${result.model} · $${result.cost.toFixed(6)}` })
      } else {
        toast.warning('Budget overrun blocked', { description: 'Oversized request rejected before it hit the provider.' })
      }
    },
    clearLogs: async () => {
      if (!client) return
      const ok = await run('clear', async () => {
        await client.clearLogs()
        return true
      })
      if (ok) toast.success('Request logs cleared')
    },
    exportLogs: async (format) => {
      if (!client) return
      await run('export', () => client.exportLogs(format, statusFilter))
    },
  }

  return <TokenGuardContext.Provider value={value}>{children}</TokenGuardContext.Provider>
}
