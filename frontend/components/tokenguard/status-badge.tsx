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
