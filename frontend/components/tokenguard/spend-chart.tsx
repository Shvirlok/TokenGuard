'use client'

import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, XAxis, YAxis } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { parseTimestamp } from '@/lib/tokenguard/format'
import { useTokenGuard } from './provider'

const chartConfig = {
  spend: { label: 'Spend ($)', color: 'var(--primary)' },
  requests: { label: 'Requests', color: 'var(--info)' },
} satisfies ChartConfig

export function SpendChart() {
  const { history } = useTokenGuard()

  const data = useMemo(
    () =>
      (history ?? []).map((p) => ({
        label: new Date(parseTimestamp(p.hour)).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
        spend: Number(p.spend_usd.toFixed(5)),
        requests: p.request_count,
      })),
    [history],
  )

  const totals = useMemo(
    () => ({
      spend: (history ?? []).reduce((s, p) => s + p.spend_usd, 0),
      requests: (history ?? []).reduce((s, p) => s + p.request_count, 0),
      blocked: (history ?? []).reduce((s, p) => s + p.blocked_count, 0),
    }),
    [history],
  )

  return (
    <section aria-labelledby="chart-title" className="flex flex-col gap-5 rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 id="chart-title" className="text-lg font-semibold tracking-tight">
            Spending & Request Velocity (24h)
          </h2>
          <p className="text-sm text-muted-foreground">Hourly cost consumption vs request volume</p>
        </div>
        <dl className="flex gap-6 text-sm">
          <div>
            <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="size-2 rounded-full bg-primary" aria-hidden="true" />
              Spend
            </dt>
            <dd className="font-mono font-semibold">${totals.spend.toFixed(4)}</dd>
          </div>
          <div>
            <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="size-2 rounded-full bg-info" aria-hidden="true" />
              Requests
            </dt>
            <dd className="font-mono font-semibold">{totals.requests}</dd>
          </div>
          <div>
            <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="size-2 rounded-full bg-destructive" aria-hidden="true" />
              Blocked
            </dt>
            <dd className="font-mono font-semibold">{totals.blocked}</dd>
          </div>
        </dl>
      </div>

      {history ? (
        <ChartContainer config={chartConfig} className="aspect-auto h-64 w-full">
          <ComposedChart data={data} margin={{ left: 4, right: 4, top: 8 }}>
            <defs>
              <linearGradient id="spendFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-spend)" stopOpacity={0.45} />
                <stop offset="100%" stopColor="var(--color-spend)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="label" tickLine={false} axisLine={false} tickMargin={8} minTickGap={24} />
            <YAxis
              yAxisId="spend"
              tickLine={false}
              axisLine={false}
              width={56}
              tickFormatter={(v: number) => `$${v.toFixed(3)}`}
            />
            <YAxis yAxisId="requests" orientation="right" tickLine={false} axisLine={false} width={28} allowDecimals={false} />
            <ChartTooltip content={<ChartTooltipContent indicator="line" />} />
            <Area
              yAxisId="spend"
              type="monotone"
              dataKey="spend"
              stroke="var(--color-spend)"
              strokeWidth={2}
              fill="url(#spendFill)"
              isAnimationActive={false}
            />
            <Line
              yAxisId="requests"
              type="monotone"
              dataKey="requests"
              stroke="var(--color-requests)"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ChartContainer>
      ) : (
        <div className="h-64 animate-pulse rounded-xl bg-muted" aria-busy="true" />
      )}
    </section>
  )
}
