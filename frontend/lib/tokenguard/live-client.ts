import { DEFAULT_PROXY_URL } from './constants'
import type { TokenGuardClient } from './types'

export async function detectProxy(): Promise<string | null> {
  const candidates = Array.from(
    new Set([process.env.NEXT_PUBLIC_TOKENGUARD_URL, '', DEFAULT_PROXY_URL].filter((c): c is string => c !== undefined)),
  )
  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500), cache: 'no-store' })
      if (!res.ok) continue
      const data = await res.json()
      if (data?.service === 'TokenGuard') return base
    } catch {
      // Proxy not reachable at this address; try the next candidate.
    }
  }
  return null
}

export function createLiveClient(base: string): TokenGuardClient {
  const call = async <T,>(path: string, body?: unknown): Promise<T> => {
    const res = await fetch(`${base}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
    })
    if (!res.ok) throw new Error(`TokenGuard API ${path} failed (${res.status})`)
    return res.json() as Promise<T>
  }
  const statusQuery = (status: string) => (status === 'all' ? '' : `&status=${status}`)

  return {
    getStats: () => call('/api/stats'),
    getRequests: async (status) =>
      (await call<{ requests: never[] }>(`/api/requests?limit=100${statusQuery(status)}`)).requests,
    getHistory: async () => (await call<{ history: never[] }>('/api/history?hours=24')).history,
    getPrices: async () => (await call<{ prices: never }>('/api/prices')).prices,
    toggleKillSwitch: async () =>
      (await call<{ kill_switch_active: boolean }>('/api/kill-switch', { source: 'web' })).kill_switch_active,
    updateConfig: async (update) => {
      await call('/api/config', { ...update, source: 'web' })
    },
    simulate: (type) => call('/api/simulate', { type }),
    clearLogs: async () => {
      await call('/api/clear-logs', { source: 'web' })
    },
    exportLogs: async (format, status) => {
      const a = document.createElement('a')
      a.href = `${base}/api/export?format=${format}${statusQuery(status)}`
      a.download = `tokenguard_requests.${format}`
      a.click()
    },
  }
}
