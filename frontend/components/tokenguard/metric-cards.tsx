'use client'

import { formatNumber, formatUsd } from '@/lib/tokenguard/format'
import { useTokenGuard } from './provider'

export function MetricCards() {
  const { stats } = useTokenGuard()

  const activeStreams = stats
    ? (stats.success_requests ?? Math.max(0, stats.total_requests - stats.blocked_requests))
    : 0
  const loopsIntercepted = stats
    ? (stats.all_time_blocked_loops ?? stats.blocked_loop_count ?? 0)
    : 0
  const budgetPreserved = stats ? (stats.saved_cost_estimate ?? 0) : 0

  return (
    <div className="flex flex-col gap-1.5 font-mono">
      <div className="grid grid-cols-3 gap-2">
        <div className="flex flex-col justify-between rounded border border-[#27272a] bg-[#101012] px-3.5 py-2.5">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-zinc-400">
            <span>Active Streams</span>
            <span className="inline-flex items-center gap-1 text-[10px] text-[#10b981]">
              <span className="size-1.5 rounded-full bg-[#10b981]" />
              PASSTHROUGH
            </span>
          </div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-zinc-100">
              {formatNumber(activeStreams)}
            </span>
            <span className="text-[11px] text-zinc-500">
              {stats ? `${formatNumber(stats.total_requests)} total calls` : '—'}
            </span>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded border border-[#27272a] bg-[#101012] px-3.5 py-2.5">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-zinc-400">
            <span>Loops Intercepted</span>
            <span className="text-[10px] text-rose-400">
              CUTOFF: {stats?.loop_threshold ?? 3}x
            </span>
          </div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-zinc-100">
              {formatNumber(loopsIntercepted)}
            </span>
            <span className="text-[11px] text-zinc-500">
              identical loops killed
            </span>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded border border-[#27272a] bg-[#101012] px-3.5 py-2.5">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-zinc-400">
            <span>Est. Budget Preserved</span>
            <span className="text-[10px] text-[#10b981]">
              SAFEGUARDED
            </span>
          </div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-[#10b981]">
              {formatUsd(budgetPreserved, 2)}
            </span>
            <span className="text-[11px] text-zinc-500">
              runaway burn prevented
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 rounded border border-[#27272a] bg-[#101012] px-3.5 py-1.5 text-[11px] text-zinc-400">
        <div className="flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-amber-400" />
          <span>RETRY CEILINGS:</span>
          <span className="font-bold text-amber-300">{formatNumber(stats?.blocked_retry_ceiling_count ?? 0)}</span>
        </div>
        <div className="h-3 w-px bg-[#27272a]" />
        <div className="flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-purple-400" />
          <span>STAGNATION:</span>
          <span className="font-bold text-purple-300">{formatNumber(stats?.blocked_stagnation_count ?? 0)}</span>
        </div>
        <div className="h-3 w-px bg-[#27272a]" />
        <div className="flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-red-400" />
          <span>SECRETS BLOCKED:</span>
          <span className="font-bold text-red-300">{formatNumber(stats?.blocked_secret_count ?? 0)}</span>
        </div>
        <div className="h-3 w-px bg-[#27272a]" />
        <div className="flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-cyan-400" />
          <span>SESSION LIMIT:</span>
          <span className="font-bold text-cyan-300">${(stats?.session_limit ?? 0).toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}

