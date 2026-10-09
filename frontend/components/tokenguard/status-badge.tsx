import { cn } from '@/lib/utils'

const STYLES: Record<string, { label: string; className: string }> = {
  success: { label: 'Success', className: 'border-success/30 bg-success/10 text-success' },
  blocked_loop: { label: 'Loop blocked', className: 'border-destructive/30 bg-destructive/10 text-destructive' },
  blocked_budget: { label: 'Over budget', className: 'border-warning/30 bg-warning/10 text-warning' },
  blocked_killswitch: { label: 'Killed', className: 'border-destructive/40 bg-destructive/15 text-destructive' },
}

export function StatusBadge({ status }: { status: string }) {
  const style = STYLES[status] ?? { label: status, className: 'border-border bg-muted text-muted-foreground' }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium',
        style.className,
      )}
    >
      <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
      {style.label}
    </span>
  )
}
