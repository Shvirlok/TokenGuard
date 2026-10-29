import { cn } from '@/lib/utils'

const STYLES: Record<string, { label: string; className: string }> = {
  success: {
    label: '200 OK',
    className: 'border-[#10b981]/50 bg-[#10b981]/10 text-[#10b981]',
  },
  blocked_loop: {
    label: 'CIRCUIT BREAKER: HARD STOP',
    className: 'border-rose-500/80 bg-rose-950/50 text-rose-300 font-bold',
  },
  blocked_tool_loop: {
    label: 'TOOL RETRY LOOP: HARD STOP',
    className: 'border-rose-500/80 bg-rose-950/60 text-rose-300 font-bold',
  },
  blocked_tool_retry_ceiling: {
    label: 'RETRY CEILING',
    className: 'border-amber-500/80 bg-amber-950/60 text-amber-300 font-bold',
  },
  blocked_stagnation: {
    label: 'STAGNATION',
    className: 'border-purple-500/80 bg-purple-950/60 text-purple-300 font-bold',
  },
  blocked_secret_leak: {
    label: 'SECRET BLOCKED',
    className: 'border-red-600/80 bg-red-950/70 text-red-300 font-bold',
  },
  blocked_stream_loop: {
    label: 'STREAM ABORT',
    className: 'border-rose-500/80 bg-rose-950/60 text-rose-300 font-bold',
  },
  blocked_session_budget: {
    label: 'SESSION BUDGET',
    className: 'border-cyan-500/80 bg-cyan-950/60 text-cyan-300 font-bold',
  },
  blocked_burn_rate: {
    label: '429 BURN RATE SPIKE',
    className: 'border-amber-500/80 bg-amber-950/60 text-amber-300 font-bold',
  },
  blocked_context_ballooning: {
    label: '429 CONTEXT RUNAWAY',
    className: 'border-rose-500/80 bg-rose-950/60 text-rose-300 font-bold',
  },
  blocked_budget: {
    label: '429 OVER BUDGET',
    className: 'border-amber-900/60 bg-amber-950/30 text-amber-400',
  },
  blocked_killswitch: {
    label: '503 KILL SWITCH',
    className: 'border-rose-900/60 bg-rose-950/30 text-rose-400',
  },
}

export function StatusBadge({ status }: { status: string }) {
  const style = STYLES[status] ?? {
    label: status.toUpperCase(),
    className: 'border-[#27272a] bg-[#141416] text-zinc-400',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 whitespace-nowrap rounded border px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase',
        style.className,
      )}
    >
      <span className="size-1 rounded-full bg-current" aria-hidden="true" />
      {style.label}
    </span>
  )
}
