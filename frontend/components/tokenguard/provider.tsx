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
  } catch {
    // Audio unavailable (autoplay policy); alerts are optional.
  }
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

  const interval = polling ? 2000 : 0
  const key = (name: string, ...extra: string[]) => (client ? ['tg', mode, baseUrl, name, ...extra] : null)

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
      if (result.simulation === 'loop') {
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
