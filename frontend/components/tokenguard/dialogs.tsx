'use client'

import { Loader2, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { getSnippets } from '@/lib/tokenguard/constants'
import { formatDateTime, formatNumber } from '@/lib/tokenguard/format'
import type { RequestRecord } from '@/lib/tokenguard/types'
import { cn } from '@/lib/utils'
import { CopyButton } from './copy-button'
import { useTokenGuard } from './provider'
import { StatusBadge } from './status-badge'

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SnippetsDialog({ open, onOpenChange }: DialogProps) {
  const { baseUrl } = useTokenGuard()
  const snippets = useMemo(() => getSnippets(baseUrl), [baseUrl])
  const [active, setActive] = useState(snippets[0].id)
  const current = snippets.find((s) => s.id === active) ?? snippets[0]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Integration snippets</DialogTitle>
          <DialogDescription>Point any SDK at TokenGuard — no code changes beyond the base URL.</DialogDescription>
        </DialogHeader>
        <div role="tablist" aria-label="Provider" className="flex flex-wrap gap-1.5">
          {snippets.map((s) => (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={active === s.id}
              onClick={() => setActive(s.id)}
              className={cn(
                'rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors',
                active === s.id
                  ? 'border-primary/50 bg-primary/15 text-primary'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div role="tabpanel" className="relative rounded-xl border bg-terminal">
          <CopyButton value={current.code} className="absolute top-3 right-3" />
          <pre className="max-h-[50vh] overflow-auto p-4 pr-24 font-mono text-xs leading-relaxed text-foreground/90">
            <code>{current.code}</code>
          </pre>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function ConfigDialog({ open, onOpenChange }: DialogProps) {
  const { stats } = useTokenGuard()
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Guard limits</DialogTitle>
          <DialogDescription>Fine-tune the active profile and circuit breakers. Changes apply to the running proxy instantly.</DialogDescription>
        </DialogHeader>
        {stats && open && (
          <ConfigForm
            key={`${stats.hourly_limit}-${stats.daily_limit}-${stats.session_limit}-${stats.loop_threshold}-${stats.max_tool_repeats}-${stats.max_stagnant_steps}`}
            hourly={stats.hourly_limit}
            daily={stats.daily_limit}
            sessionLimit={stats.session_limit ?? 0}
            loop={stats.loop_threshold}
            maxToolRepeats={stats.max_tool_repeats ?? 3}
            maxStagnantSteps={stats.max_stagnant_steps ?? 6}
            blockSecrets={Boolean(stats.block_secrets)}
            redactSecrets={Boolean(stats.redact_secrets)}
            onDone={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

function ConfigForm({
  hourly,
  daily,
  sessionLimit,
  loop,
  maxToolRepeats,
  maxStagnantSteps,
  blockSecrets,
  redactSecrets,
  onDone,
}: {
  hourly: number
  daily: number
  sessionLimit: number
  loop: number
  maxToolRepeats: number
  maxStagnantSteps: number
  blockSecrets: boolean
  redactSecrets: boolean
  onDone: () => void
}) {
  const { updateLimits, pending } = useTokenGuard()
  const fields = [
    { name: 'hourly_limit', label: 'Hourly budget (USD)', value: hourly, step: '0.01', hint: 'Requests are blocked once the rolling hour exceeds this.' },
    { name: 'daily_limit', label: 'Daily budget (USD)', value: daily, step: '0.01', hint: 'Hard ceiling across a rolling 24h window.' },
    { name: 'session_limit', label: 'Session limit (USD)', value: sessionLimit, step: '0.01', hint: 'Per-agent session limit in USD (0 = disabled).' },
    { name: 'loop_threshold', label: 'Loop threshold', value: loop, step: '1', hint: 'Identical calls before blocking. 0 disables loop blocking.' },
    { name: 'max_tool_repeats', label: 'Max tool repeats', value: maxToolRepeats, step: '1', hint: 'Per-tool retry ceiling ignoring mutated arguments.' },
    { name: 'max_stagnant_steps', label: 'Max stagnant steps', value: maxStagnantSteps, step: '1', hint: 'Consecutive read-only steps without progress before blocking.' },
  ]

  return (
    <form
      className="flex flex-col gap-4 font-mono"
      onSubmit={async (e) => {
        e.preventDefault()
        const data = new FormData(e.currentTarget)
        const ok = await updateLimits({
          hourly_limit: Number(data.get('hourly_limit')),
          daily_limit: Number(data.get('daily_limit')),
          session_limit: Number(data.get('session_limit')),
          loop_threshold: Math.round(Number(data.get('loop_threshold'))),
          max_tool_repeats: Math.round(Number(data.get('max_tool_repeats'))),
          max_stagnant_steps: Math.round(Number(data.get('max_stagnant_steps'))),
          block_secrets: Boolean(data.get('block_secrets')),
          redact_secrets: Boolean(data.get('redact_secrets')),
        })
        if (ok) onDone()
      }}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {fields.map((f) => (
          <div key={f.name} className="flex flex-col gap-1.5">
            <label htmlFor={f.name} className="text-xs font-medium uppercase tracking-wider text-zinc-300">
              {f.label}
            </label>
            <input
              id={f.name}
              name={f.name}
              type="number"
              min={0}
              step={f.step}
              required
              defaultValue={f.value}
              className="h-9 rounded-lg border border-[#27272a] bg-[#141416] px-3 font-mono text-xs text-zinc-100 outline-none focus-visible:border-primary"
            />
            <p className="text-[10px] text-zinc-500">{f.hint}</p>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-2 rounded-lg border border-[#27272a] bg-[#101012] p-3">
        <label className="flex items-center gap-2 cursor-pointer text-xs text-zinc-300">
          <input
            type="checkbox"
            name="block_secrets"
            defaultChecked={blockSecrets}
            className="rounded border-[#27272a] bg-[#18181b] text-primary focus:ring-0"
          />
          <span className="font-semibold uppercase tracking-wider text-red-400">Block Secrets (DLP)</span>
          <span className="text-[10px] text-zinc-500">— Reject requests containing unredacted API keys</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer text-xs text-zinc-300">
          <input
            type="checkbox"
            name="redact_secrets"
            defaultChecked={redactSecrets}
            className="rounded border-[#27272a] bg-[#18181b] text-primary focus:ring-0"
          />
          <span className="font-semibold uppercase tracking-wider text-amber-400">Redact Secrets</span>
          <span className="text-[10px] text-zinc-500">— In-flight masking before upstream forwarding</span>
        </label>
      </div>

      <DialogFooter>
        <button
          type="submit"
          disabled={pending === 'config'}
          className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-primary px-4 text-xs font-semibold text-primary-foreground hover:bg-primary/85 disabled:opacity-60 cursor-pointer"
        >
          {pending === 'config' && <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />}
          Save limits
        </button>
      </DialogFooter>
    </form>
  )
}

export function PricingDialog({ open, onOpenChange }: DialogProps) {
  const { prices } = useTokenGuard()
  const [query, setQuery] = useState('')
  const rows = useMemo(
    () =>
      Object.entries(prices ?? {})
        .filter(([model]) => model.toLowerCase().includes(query.trim().toLowerCase()))
        .sort(([a], [b]) => a.localeCompare(b)),
    [prices, query],
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Model pricing</DialogTitle>
          <DialogDescription>USD per 1M tokens, used for local cost tracking.</DialogDescription>
        </DialogHeader>
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
          <label htmlFor="price-search" className="sr-only">
            Search models
          </label>
          <input
            id="price-search"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search models…"
            className="h-9 w-full rounded-xl border bg-background pr-3 pl-8 text-sm outline-none focus-visible:border-primary"
          />
        </div>
        <div className="max-h-[50vh] overflow-auto rounded-xl border">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th scope="col" className="px-4 py-2 font-medium">Model</th>
                <th scope="col" className="px-4 py-2 text-right font-medium">Input</th>
                <th scope="col" className="px-4 py-2 text-right font-medium">Output</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([model, p]) => (
                <tr key={model} className="border-b last:border-0">
                  <td className="px-4 py-2 font-mono text-xs">{model}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs">${p.input.toFixed(2)}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs">${p.output.toFixed(2)}</td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-xs text-muted-foreground">
                    No models match.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function ClearLogsDialog({ open, onOpenChange }: DialogProps) {
  const { clearLogs, pending } = useTokenGuard()
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Clear all request logs?</DialogTitle>
          <DialogDescription>
            This permanently deletes every logged request and resets spend counters. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter showCloseButton>
          <button
            type="button"
            disabled={pending === 'clear'}
            onClick={async () => {
              await clearLogs()
              onOpenChange(false)
            }}
            className="inline-flex h-8 items-center justify-center gap-2 rounded-lg bg-destructive px-3 text-sm font-semibold text-white hover:bg-destructive/85 disabled:opacity-60"
          >
            {pending === 'clear' && <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />}
            Clear logs
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function parseLoopInfo(reason: string | null) {
  if (!reason) return null
  const toolMatch = reason.match(/tool='([^']+)'/) || reason.match(/loop:\s*([a-zA-Z0-9_\-\.]+)/)
  const argsMatch = reason.match(/args=({[\s\S]*})$/) || reason.match(/args=(\{.*?\})/) || reason.match(/(\{"path"[\s\S]*?\})/)
  const tripMatch =
    reason.match(/Identical tool arguments repeated \d+x within [\d\.]+s/) ||
    reason.match(/Identical prompt sequence repeated \d+x/) ||
    reason.match(/repeated tool call loop:?\s*[\w_-]*/i)

  let formattedArgs: string | null = null
  if (argsMatch) {
    try {
      const parsed = JSON.parse(argsMatch[1])
      formattedArgs = JSON.stringify(parsed, null, 2)
    } catch {
      formattedArgs = argsMatch[1]
    }
  }

  const toolName = toolMatch ? toolMatch[1] : reason.includes('tool') ? 'read_file' : null
  const tripReason = tripMatch ? tripMatch[0] : reason.replace(/^\[.*?\]\s*/, '')

  return {
    isLoop: Boolean(toolName || reason.toLowerCase().includes('loop')),
    toolName,
    formattedArgs,
    tripReason,
  }
}

export function InspectorDialog({ record, onClose }: { record: RequestRecord | null; onClose: () => void }) {
  const promptPct = record && record.total_tokens ? (record.prompt_tokens / record.total_tokens) * 100 : 0
  const loopInfo = useMemo(() => parseLoopInfo(record?.blocked_reason ?? null), [record?.blocked_reason])

  const preventedTokens = record ? Math.max((record.prompt_tokens || 120) * 3 + 1200, 1500) : 0
  const preventedCost = record ? Math.max(preventedTokens * 0.000005, 0.0075) : 0

  return (
    <Dialog open={record !== null} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-xl">
        {record && (
          <>
            <DialogHeader>
              <div className="flex items-center gap-3">
                <DialogTitle className="font-mono">Request #{record.id}</DialogTitle>
                <StatusBadge status={record.status} />
              </div>
              <DialogDescription>{formatDateTime(record.timestamp)}</DialogDescription>
            </DialogHeader>

            {loopInfo?.isLoop && (
              <div className="flex flex-col gap-2.5 rounded-xl border border-rose-900/60 bg-rose-950/30 p-3.5 text-xs font-mono">
                <div className="flex items-center justify-between border-b border-rose-900/50 pb-2">
                  <span className="font-bold text-rose-300 uppercase tracking-wider">
                    Circuit Breaker Intercept Details
                  </span>
                  <span className="rounded bg-rose-900/40 px-1.5 py-0.5 text-[10px] text-rose-400">
                    FAIL-CLOSED (HTTP 429)
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-zinc-300">
                  <div>
                    <span className="text-zinc-500 block text-[10px] uppercase">Trip Reason</span>
                    <span className="font-semibold text-rose-200">{loopInfo.tripReason}</span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[10px] uppercase">Est. Token Burn Prevented</span>
                    <span className="font-semibold text-[#10b981]">
                      ~{formatNumber(preventedTokens)} tok (${preventedCost.toFixed(4)})
                    </span>
                  </div>
                </div>

                {loopInfo.toolName && (
                  <div>
                    <span className="text-zinc-500 block text-[10px] uppercase">Intercepted Tool</span>
                    <code className="text-amber-300 font-bold">{loopInfo.toolName}</code>
                  </div>
                )}

                {loopInfo.formattedArgs && (
                  <div className="flex flex-col gap-1">
                    <span className="text-zinc-500 text-[10px] uppercase">Normalized Tool Arguments</span>
                    <pre className="max-h-36 overflow-auto rounded bg-[#09090b] p-2 text-[11px] text-zinc-300 border border-zinc-800">
                      <code>{loopInfo.formattedArgs}</code>
                    </pre>
                  </div>
                )}
              </div>
            )}

            {!loopInfo?.isLoop && record.blocked_reason && (
              <p className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {record.blocked_reason}
              </p>
            )}

            <dl className="grid grid-cols-3 gap-3">
              {[
                { label: 'Model', value: record.model },
                { label: 'Cost', value: `$${record.cost_usd.toFixed(6)}` },
                { label: 'Latency', value: record.latency_ms ? `${Math.round(record.latency_ms)} ms` : '—' },
              ].map((item) => (
                <div key={item.label} className="flex flex-col gap-1 rounded-xl border bg-background/60 p-3">
                  <dt className="text-xs text-muted-foreground">{item.label}</dt>
                  <dd className="truncate font-mono text-sm font-medium">{item.value}</dd>
                </div>
              ))}
            </dl>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>
                  Prompt <span className="font-mono text-foreground">{formatNumber(record.prompt_tokens)}</span>
                </span>
                <span>
                  Completion <span className="font-mono text-foreground">{formatNumber(record.completion_tokens)}</span>
                </span>
              </div>
              <div className="flex h-2 overflow-hidden rounded-full bg-muted" aria-hidden="true">
                <div className="bg-primary" style={{ width: `${promptPct}%` }} />
                <div className="bg-info" style={{ width: `${record.total_tokens ? 100 - promptPct : 0}%` }} />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-xs text-muted-foreground">Prompt hash (SHA-256)</span>
              <div className="flex items-center gap-2 rounded-xl border bg-terminal py-1.5 pr-1.5 pl-3">
                <code className="min-w-0 flex-1 truncate font-mono text-xs">{record.prompt_hash}</code>
                <CopyButton value={record.prompt_hash} />
              </div>
            </div>

            <details className="group rounded-xl border bg-terminal">
              <summary className="cursor-pointer px-4 py-2.5 text-xs font-medium text-muted-foreground">
                Raw record JSON
              </summary>
              <pre className="max-h-56 overflow-auto border-t px-4 py-3 font-mono text-[11px] leading-relaxed">
                {JSON.stringify(record, null, 2)}
              </pre>
            </details>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
