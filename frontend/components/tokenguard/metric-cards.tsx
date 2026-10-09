'use client'

import { CircleCheck, CircleDollarSign, Coins, Zap } from 'lucide-react'
import { formatNumber, formatSaved, formatUsd } from '@/lib/tokenguard/format'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'

function MetricCard({
  title,
  icon,
  iconClassName,
  children,
}: {
  title: string
  icon: React.ReactNode
  iconClassName: string
  children: React.ReactNode
}) {
  return (
    <article className="flex flex-col gap-4 rounded-2xl border bg-card p-5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm text-muted-foreground">{title}</h3>
        <span className={cn('flex size-9 items-center justify-center rounded-xl', iconClassName)}>{icon}</span>
      </div>
      {children}
    </article>
  )
}

function Skeleton({ className }: { className?: string }) {
  return <span className={cn('block animate-pulse rounded-md bg-muted', className)} />
}

export function MetricCards() {
  const { stats } = useTokenGuard()

  if (!stats) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-busy="true">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="flex flex-col gap-4 rounded-2xl border bg-card p-5">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-9 w-40" />
            <Skeleton className="h-3 w-full" />
          </div>
        ))}
      </div>
    )
  }

  const pct = Math.min(100, Math.round((stats.total_spent / (stats.hourly_limit || 1)) * 100))
  const barColor = pct >= 90 ? 'bg-destructive' : pct >= 70 ? 'bg-warning' : 'bg-primary'

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        title="Hourly Spend / Budget"
        icon={<CircleDollarSign className="size-4" aria-hidden="true" />}
        iconClassName="bg-primary/15 text-primary"
      >
        <p className="flex items-baseline gap-1.5">
          <span className="font-mono text-3xl font-semibold tracking-tight">{formatUsd(stats.total_spent)}</span>
          <span className="text-sm text-muted-foreground">/ {formatUsd(stats.hourly_limit, 2)}</span>
        </p>
        <div
          role="progressbar"
          aria-label="Hourly budget used"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          className="h-2 overflow-hidden rounded-full bg-muted"
        >
          <div className={cn('h-full rounded-full transition-all duration-500', barColor)} style={{ width: `${Math.max(pct, 1)}%` }} />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{pct}% of hourly limit</span>
          <span>24h: {formatUsd(stats.current_daily_spend)}</span>
        </div>
      </MetricCard>

      <MetricCard
        title="Estimated Money Saved"
        icon={<CircleCheck className="size-4" aria-hidden="true" />}
        iconClassName="bg-success/15 text-success"
      >
        <p className="font-mono text-3xl font-semibold tracking-tight text-success">
          {formatSaved(stats.saved_cost_estimate)}
        </p>
        <p className="text-xs text-muted-foreground">Saved by intercepting runaway loops & limits</p>
        <div className="flex flex-wrap gap-2">
          <span className="rounded-md border border-destructive/30 bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive">
            {stats.all_time_blocked_loops} Loops Caught
          </span>
          <span className="rounded-md border border-warning/30 bg-warning/10 px-2 py-0.5 text-xs font-medium text-warning">
            {stats.all_time_blocked_budget ?? stats.blocked_budget_count} Budget Overruns
          </span>
        </div>
      </MetricCard>

      <MetricCard
        title="Requests (1h) & Latency"
        icon={<Zap className="size-4" aria-hidden="true" />}
        iconClassName="bg-info/15 text-info"
      >
        <div className="flex items-end justify-between gap-2">
          <p className="flex items-baseline gap-1.5">
            <span className="font-mono text-3xl font-semibold tracking-tight">{stats.total_requests}</span>
            <span className="text-sm text-muted-foreground">total reqs</span>
          </p>
          <p className="flex flex-col items-end">
            <span className="font-mono text-xl font-semibold text-info">{Math.round(stats.avg_latency_ms)} ms</span>
            <span className="text-xs text-muted-foreground">avg latency</span>
          </p>
        </div>
        <div className="mt-auto flex justify-between border-t pt-3 text-xs">
          <span className="text-success">{stats.success_requests} successful</span>
          <span className="text-destructive">{stats.blocked_requests} blocked</span>
        </div>
      </MetricCard>

      <MetricCard
        title="Total Tokens Processed"
        icon={<Coins className="size-4" aria-hidden="true" />}
        iconClassName="bg-primary/15 text-primary"
      >
        <p className="font-mono text-3xl font-semibold tracking-tight">{formatNumber(stats.total_tokens)}</p>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>
            Prompt: <span className="font-mono text-foreground">{formatNumber(stats.total_prompt_tokens)}</span>
          </span>
          <span>
            Completion: <span className="font-mono text-foreground">{formatNumber(stats.total_completion_tokens)}</span>
          </span>
        </div>
        <p className="mt-auto border-t pt-3 text-xs text-muted-foreground">
          All-time: <span className="font-mono text-foreground">{formatNumber(stats.all_time_tokens)}</span> tokens
        </p>
      </MetricCard>
    </div>
  )
}
