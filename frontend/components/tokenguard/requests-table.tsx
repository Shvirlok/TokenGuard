'use client'

import { Pause, Play, Search, Trash2, Zap } from 'lucide-react'
import { useMemo, useState } from 'react'
import { formatNumber, formatTime } from '@/lib/tokenguard/format'
import type { RequestRecord, StatusFilter } from '@/lib/tokenguard/types'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'

interface RequestsTableProps {
  onInspect: (record: RequestRecord) => void
  onClearLogs?: () => void
}

export function RequestsTable({ onInspect, onClearLogs }: RequestsTableProps) {
  const { requests, statusFilter, setStatusFilter, polling, setPolling, simulate, pending } = useTokenGuard()
  const [query, setQuery] = useState('')

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return requests ?? []
    return (requests ?? []).filter((r) =>
      [r.model, r.status, r.prompt_hash, r.blocked_reason ?? ''].some((f) => f.toLowerCase().includes(q)),
    )
  }, [requests, query])

  return (
    <section
      aria-label="Traffic Stream Logs"
      className="flex h-full min-h-0 flex-col rounded border border-[#27272a] bg-[#101012] font-mono"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[#27272a] px-3 py-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-zinc-200">
            <span
              className={cn(
                'size-2 rounded-full',
                polling ? 'bg-[#10b981] animate-pulse' : 'bg-zinc-600'
              )}
            />
            <span>Traffic Stream</span>
          </div>

          <div className="h-3 w-px bg-[#27272a]" />

          <button
            type="button"
            onClick={() => simulate('tool_loop')}
            disabled={pending === 'sim:tool_loop' || pending === 'sim:loop'}
            title="Simulate repeated tool-call loop to trigger the circuit breaker"
            className="inline-flex h-6 items-center gap-1.5 rounded border border-[#3f3f46] bg-[#18181b] px-2 text-[10px] font-semibold tracking-wider uppercase text-zinc-200 transition-colors hover:border-amber-700 hover:bg-amber-950/30 hover:text-amber-400 disabled:opacity-50 cursor-pointer"
          >
            <Zap className="size-2.5 text-amber-400" />
            <span>{pending === 'sim:tool_loop' || pending === 'sim:loop' ? 'RUNNING…' : '+ TRIGGER TEST LOOP'}</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="filter..."
              className="h-6 w-28 rounded border border-[#27272a] bg-[#09090b] pl-6 pr-2 text-[11px] text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none sm:w-36"
            />
          </div>

          <div className="flex rounded border border-[#27272a] bg-[#09090b] p-0.5 text-[10px]">
            {(['all', 'success', 'blocked'] as StatusFilter[]).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setStatusFilter(f)}
                className={cn(
                  'rounded px-1.5 py-0.5 uppercase transition-colors cursor-pointer',
                  statusFilter === f
                    ? 'bg-[#27272a] text-zinc-100 font-semibold'
                    : 'text-zinc-500 hover:text-zinc-300'
                )}
              >
                {f}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setPolling(!polling)}
            title={polling ? 'Pause live stream' : 'Resume live stream'}
            className={cn(
              'flex size-6 items-center justify-center rounded border transition-colors cursor-pointer',
              polling
                ? 'border-[#27272a] bg-[#141416] text-zinc-400 hover:text-zinc-200'
                : 'border-amber-800 bg-amber-950/30 text-amber-400'
            )}
          >
            {polling ? <Pause className="size-2.5" /> : <Play className="size-2.5" />}
          </button>

          {onClearLogs && (
            <button
              type="button"
              onClick={onClearLogs}
              title="Clear logs"
              className="flex size-6 items-center justify-center rounded border border-[#27272a] bg-[#141416] text-zinc-500 hover:border-rose-900 hover:text-rose-400 cursor-pointer"
            >
              <Trash2 className="size-2.5" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <table className="w-full text-left text-[11px] border-collapse">
          <thead className="sticky top-0 z-10 border-b border-[#27272a] bg-[#141416] text-[10px] uppercase text-zinc-500 tracking-wider">
            <tr>
              <th className="py-1.5 pl-3 pr-2 font-medium">Time</th>
              <th className="py-1.5 px-2 font-medium">Endpoint</th>
              <th className="py-1.5 px-2 font-medium">Status</th>
              <th className="py-1.5 px-2 text-right font-medium">Tokens</th>
              <th className="py-1.5 pr-3 pl-2 text-right font-medium">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#27272a]/50">
            {(!requests || requests.length === 0) && (
              <tr>
                <td colSpan={5} className="py-12 text-center text-zinc-500 text-xs">
                  <span>No traffic logged. Click [TRIGGER TEST LOOP] or route OpenAI calls through 127.0.0.1:8080/v1</span>
                </td>
              </tr>
            )}

            {rows.map((r) => {
              const isToolLoopBlocked =
                r.status === 'blocked_tool_loop' ||
                (Boolean(r.blocked_reason?.toLowerCase().includes('tool')) &&
                  Boolean(r.blocked_reason?.toLowerCase().includes('loop')))
              const isLoopBlocked =
                r.status === 'blocked_loop' ||
                (!isToolLoopBlocked && Boolean(r.blocked_reason?.toLowerCase().includes('loop')))
              const isBudgetBlocked = r.status === 'blocked_budget'
              const isKilled = r.status === 'blocked_killswitch'
              const isSuccess = r.status === 'success'

              return (
                <tr
                  key={r.id}
                  onClick={() => onInspect(r)}
                  className={cn(
                    'group cursor-pointer transition-colors hover:bg-[#18181b]/80 select-none',
                    (isToolLoopBlocked || isLoopBlocked) && 'bg-rose-950/20 hover:bg-rose-950/30'
                  )}
                >
                  <td className="py-1.5 pl-3 pr-2 text-zinc-500 whitespace-nowrap">
                    {formatTime(r.timestamp)}
                  </td>

                  <td className="py-1.5 px-2 font-mono text-zinc-300">
                    <div className="flex items-center gap-1.5">
                      <span className="text-zinc-300 font-medium">/v1/chat/completions</span>
                      <span className="text-[10px] text-zinc-500">({r.model})</span>
                    </div>
                  </td>

                  <td className="py-1.5 px-2 whitespace-nowrap">
                    {isToolLoopBlocked ? (
                      <span className="inline-flex items-center gap-1.5 rounded border border-rose-500/80 bg-rose-950/60 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-rose-300">
                        <span className="size-1.5 rounded-full bg-rose-500 animate-pulse" />
                        [TOOL RETRY LOOP: HARD STOP]
                      </span>
                    ) : isLoopBlocked ? (
                      <span className="inline-flex items-center gap-1.5 rounded border border-rose-500/80 bg-rose-950/60 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-rose-300">
                        <span className="size-1.5 rounded-full bg-rose-500 animate-pulse" />
                        [CIRCUIT BREAKER: HARD STOP]
                      </span>
                    ) : isBudgetBlocked ? (
                      <span className="inline-flex items-center gap-1 rounded border border-amber-900/60 bg-amber-950/30 px-1.5 py-0.5 text-[10px] text-amber-400">
                        429 OVER BUDGET
                      </span>
                    ) : isKilled ? (
                      <span className="inline-flex items-center gap-1 rounded border border-rose-900/60 bg-rose-950/30 px-1.5 py-0.5 text-[10px] text-rose-400">
                        503 KILL SWITCH
                      </span>
                    ) : isSuccess ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-[#10b981] font-semibold">
                        <span className="size-1 rounded-full bg-[#10b981]" />
                        200 OK
                      </span>
                    ) : (
                      <span className="text-zinc-400 text-[10px]">{r.status}</span>
                    )}
                  </td>

                  <td className="py-1.5 px-2 text-right text-zinc-300 whitespace-nowrap">
                    {r.total_tokens ? `${formatNumber(r.total_tokens)} tok` : '—'}
                  </td>

                  <td className="py-1.5 pr-3 pl-2 text-right text-zinc-500 whitespace-nowrap">
                    {r.latency_ms ? `${Math.round(r.latency_ms)}ms` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-[#27272a] px-3 py-1 text-[10px] text-zinc-500">
        <span>
          Showing {rows.length} of {requests?.length ?? 0} events
        </span>
        <span className="text-zinc-600">click row to inspect</span>
      </div>
    </section>
  )
}

