'use client'

import { ChevronRight, Download, FileJson, FileSpreadsheet, Inbox, Pause, Play, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { formatNumber, formatTime } from '@/lib/tokenguard/format'
import type { RequestRecord, StatusFilter } from '@/lib/tokenguard/types'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'
import { StatusBadge } from './status-badge'

const FILTERS: { value: StatusFilter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'success', label: 'Success' },
  { value: 'blocked', label: 'Blocked' },
]

interface RequestsTableProps {
  onInspect: (record: RequestRecord) => void
  onClearLogs: () => void
}

export function RequestsTable({ onInspect, onClearLogs }: RequestsTableProps) {
  const { requests, statusFilter, setStatusFilter, polling, setPolling, exportLogs } = useTokenGuard()
  const [query, setQuery] = useState('')

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return requests ?? []
    return (requests ?? []).filter((r) =>
      [r.model, r.status, r.prompt_hash, r.blocked_reason ?? ''].some((f) => f.toLowerCase().includes(q)),
    )
  }, [requests, query])

  return (
    <section aria-labelledby="requests-title" className="flex flex-col rounded-2xl border bg-card">
      <div className="flex flex-col gap-4 border-b p-5 sm:p-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 id="requests-title" className="text-lg font-semibold tracking-tight">
            Live Request Stream
          </h2>
          <p className="text-sm text-muted-foreground">Every call routed through the proxy, newest first</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search
              className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <label htmlFor="request-search" className="sr-only">
              Search requests
            </label>
            <input
              id="request-search"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search model, hash, reason…"
              className="h-9 w-56 rounded-xl border bg-background pr-3 pl-8 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-primary"
            />
          </div>

          <div role="radiogroup" aria-label="Filter by status" className="flex rounded-xl border bg-background p-0.5">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                type="button"
                role="radio"
                aria-checked={statusFilter === f.value}
                onClick={() => setStatusFilter(f.value)}
                className={cn(
                  'h-8 rounded-lg px-3 text-xs font-medium transition-colors',
                  statusFilter === f.value
                    ? 'bg-secondary text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {f.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setPolling(!polling)}
            aria-pressed={polling}
            className={cn(
              'flex h-9 items-center gap-1.5 rounded-xl border px-3 text-xs font-medium transition-colors',
              polling ? 'border-success/40 bg-success/10 text-success' : 'border-warning/40 bg-warning/10 text-warning',
            )}
          >
            {polling ? <Pause className="size-3.5" aria-hidden="true" /> : <Play className="size-3.5" aria-hidden="true" />}
            {polling ? 'Live' : 'Paused'}
          </button>

          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <button
                  type="button"
                  className="flex h-9 items-center gap-1.5 rounded-xl border bg-background px-3 text-xs font-medium hover:bg-muted"
                />
              }
            >
              <Download className="size-3.5" aria-hidden="true" />
              Export
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => exportLogs('csv')}>
                <FileSpreadsheet aria-hidden="true" />
                Export as CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => exportLogs('json')}>
                <FileJson aria-hidden="true" />
                Export as JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <button
            type="button"
            onClick={onClearLogs}
            aria-label="Clear request logs"
            className="flex size-9 items-center justify-center rounded-xl border bg-background text-muted-foreground transition-colors hover:border-destructive/50 hover:text-destructive"
          >
            <Trash2 className="size-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th scope="col" className="px-5 py-3 font-medium sm:px-6">Time</th>
              <th scope="col" className="px-3 py-3 font-medium">Model</th>
              <th scope="col" className="px-3 py-3 font-medium">Status</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Tokens</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Cost</th>
              <th scope="col" className="px-3 py-3 text-right font-medium">Latency</th>
              <th scope="col" className="px-3 py-3 font-medium">Prompt hash</th>
              <th scope="col" className="px-5 py-3 sm:px-6">
                <span className="sr-only">Inspect</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {!requests &&
              Array.from({ length: 6 }, (_, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td colSpan={8} className="px-6 py-3.5">
                    <span className="block h-4 animate-pulse rounded bg-muted" />
                  </td>
                </tr>
              ))}
            {requests && rows.length === 0 && (
              <tr>
                <td colSpan={8} className="px-6 py-16 text-center">
                  <div className="flex flex-col items-center gap-2 text-muted-foreground">
                    <Inbox className="size-6" aria-hidden="true" />
                    <p className="text-sm font-medium text-foreground">No requests yet</p>
                    <p className="text-xs">Run a simulation or point your agent at the proxy to see traffic here.</p>
                  </div>
                </td>
              </tr>
            )}
            {rows.map((r) => (
              <tr
                key={r.id}
                onClick={() => onInspect(r)}
                className={cn(
                  'animate-row-in group cursor-pointer border-b transition-colors last:border-0 hover:bg-muted/50',
                  r.status !== 'success' && 'bg-destructive/[0.03]',
                )}
              >
                <td className="px-5 py-3 font-mono text-xs text-muted-foreground sm:px-6">{formatTime(r.timestamp)}</td>
                <td className="px-3 py-3 font-medium">{r.model}</td>
                <td className="px-3 py-3">
                  <StatusBadge status={r.status} />
                </td>
                <td className="px-3 py-3 text-right font-mono text-xs">{formatNumber(r.total_tokens)}</td>
                <td className="px-3 py-3 text-right font-mono text-xs">${r.cost_usd.toFixed(6)}</td>
                <td className="px-3 py-3 text-right font-mono text-xs text-muted-foreground">
                  {r.latency_ms ? `${Math.round(r.latency_ms)} ms` : '—'}
                </td>
                <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{r.prompt_hash.slice(0, 12)}</td>
                <td className="px-5 py-3 text-right sm:px-6">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      onInspect(r)
                    }}
                    aria-label={`Inspect request ${r.id}`}
                    className="inline-flex size-7 items-center justify-center rounded-lg text-muted-foreground group-hover:bg-background group-hover:text-foreground"
                  >
                    <ChevronRight className="size-4" aria-hidden="true" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {requests && requests.length > 0 && (
        <p className="border-t px-5 py-3 text-xs text-muted-foreground sm:px-6">
          Showing {rows.length} of {requests.length} most recent requests
        </p>
      )}
    </section>
  )
}
