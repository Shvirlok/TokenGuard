'use client'

import { ArrowRight, Check, Code2, Loader2, ReceiptText, SlidersHorizontal, Zap } from 'lucide-react'
import { DEFAULT_PROXY_URL, PROFILES, type ProfileDefinition } from '@/lib/tokenguard/constants'
import { cn } from '@/lib/utils'
import { CopyButton } from './copy-button'
import { useTokenGuard } from './provider'

const ACCENTS: Record<ProfileDefinition['accent'], { dot: string; text: string; active: string; tag: string; cta: string }> = {
  success: {
    dot: 'bg-success',
    text: 'text-success',
    active: 'border-success/70 shadow-[0_0_40px_-12px] shadow-success/50',
    tag: 'border-success/30 bg-success/10 text-success',
    cta: 'border-success/50 bg-success/15 text-success',
  },
  warning: {
    dot: 'bg-warning',
    text: 'text-warning',
    active: 'border-warning/70 shadow-[0_0_40px_-12px] shadow-warning/50',
    tag: 'border-warning/30 bg-warning/10 text-warning',
    cta: 'border-warning/50 bg-warning/15 text-warning',
  },
  info: {
    dot: 'bg-info',
    text: 'text-info',
    active: 'border-info/70 shadow-[0_0_40px_-12px] shadow-info/50',
    tag: 'border-info/30 bg-info/10 text-info',
    cta: 'border-info/50 bg-info/15 text-info',
  },
}

function StepLabel({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <h3 className="flex items-center gap-3 text-sm font-semibold uppercase tracking-wider">
      <span className="flex size-7 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
        {n}
      </span>
      {children}
    </h3>
  )
}

interface ProtectionPanelProps {
  onOpenSnippets: () => void
  onOpenConfig: () => void
  onOpenPricing: () => void
}

export function ProtectionPanel({ onOpenSnippets, onOpenConfig, onOpenPricing }: ProtectionPanelProps) {
  const { stats, pending, mode, baseUrl, simulate, setProfile } = useTokenGuard()
  const proxyUrl = `${baseUrl || DEFAULT_PROXY_URL}/v1`
  const command = 'tokenguard run python my_agent.py'

  return (
    <section
      aria-labelledby="protection-title"
      className="relative overflow-hidden rounded-3xl border bg-card p-5 sm:p-8"
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-40 left-1/2 h-80 w-[60rem] -translate-x-1/2 rounded-full bg-primary/15 blur-3xl"
      />
      <div className="relative flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-primary/40 bg-primary/10 px-3 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-primary">
              Guard profiles
            </span>
            <h2 id="protection-title" className="text-2xl font-bold tracking-tight text-balance sm:text-3xl">
              Active Protection & Controls
            </h2>
          </div>
          <p className="max-w-xl text-sm text-muted-foreground text-pretty">
            Choose your active guard profile below, or launch protected scripts with zero manual config.
          </p>
        </div>
        <button
          type="button"
          onClick={() => simulate('loop')}
          disabled={mode === 'connecting' || pending === 'sim:loop'}
          className="inline-flex h-11 shrink-0 items-center justify-center gap-2 rounded-xl bg-linear-to-r from-primary to-info px-5 text-sm font-semibold text-white shadow-lg shadow-primary/30 transition-transform hover:-translate-y-0.5 disabled:translate-y-0 disabled:opacity-60"
        >
          {pending === 'sim:loop' ? (
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          ) : (
            <Zap className="size-4" aria-hidden="true" />
          )}
          Run Instant Simulation
          <span className="font-normal opacity-80">(no API key needed)</span>
        </button>
      </div>

      <div className="relative my-6 h-px bg-border sm:my-8" />

      <div className="relative grid gap-8 lg:grid-cols-5">
        <div className="flex flex-col gap-4 lg:col-span-3">
          <StepLabel n={1}>Choose guard profile</StepLabel>
          <div role="radiogroup" aria-label="Guard profile" className="grid gap-3 sm:grid-cols-3">
            {PROFILES.map((profile) => {
              const accent = ACCENTS[profile.accent]
              const active = stats?.active_profile === profile.id
              const loading = pending === `profile:${profile.id}`
              return (
                <button
                  key={profile.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  disabled={!stats || loading}
                  onClick={() => !active && setProfile(profile.id)}
                  className={cn(
                    'group flex flex-col gap-3 rounded-2xl border-2 bg-background/60 p-4 text-left transition-all',
                    active ? accent.active : 'border-border hover:border-muted-foreground/40',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className={cn('size-2.5 rounded-full', accent.dot)} aria-hidden="true" />
                    {active && (
                      <span className={cn('flex items-center gap-1 text-xs font-semibold', accent.text)}>
                        <Check className="size-3.5" aria-hidden="true" />
                        Active
                      </span>
                    )}
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="font-semibold">{profile.name}</span>
                    <span className="text-xs leading-relaxed text-muted-foreground">{profile.description}</span>
                  </div>
                  <span className={cn('w-fit rounded-md border px-2 py-0.5 font-mono text-[10px]', accent.tag)}>
                    {profile.tag}
                  </span>
                  <span
                    className={cn(
                      'mt-auto flex h-9 items-center justify-center gap-1.5 rounded-lg border text-xs font-semibold transition-colors',
                      active ? accent.cta : 'bg-secondary text-secondary-foreground group-hover:bg-muted',
                    )}
                  >
                    {loading ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
                    ) : active ? (
                      <Check className="size-3.5" aria-hidden="true" />
                    ) : null}
                    {active ? 'Active Plan' : 'Switch to Plan'}
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        <div className="flex flex-col gap-4 lg:col-span-2">
          <StepLabel n={2}>Launch protected</StepLabel>
          <div className="flex flex-1 flex-col gap-4 rounded-2xl border bg-terminal p-4 sm:p-5">
            <p className="text-sm text-muted-foreground">Run your existing Python agent through TokenGuard:</p>
            <div className="flex items-center gap-3 rounded-xl border bg-background/50 py-2 pr-2 pl-4">
              <code className="min-w-0 flex-1 truncate font-mono text-sm text-success">
                <span className="text-muted-foreground select-none">$ </span>
                {command}
              </code>
              <CopyButton value={command} />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>
                Or configure proxy: <code className="font-mono text-primary">{proxyUrl}</code>
              </span>
              <button
                type="button"
                onClick={onOpenSnippets}
                className="inline-flex items-center gap-1 font-medium text-primary underline-offset-4 hover:underline"
              >
                More snippets
                <ArrowRight className="size-3" aria-hidden="true" />
              </button>
            </div>
            <div className="mt-auto grid grid-cols-3 gap-2 border-t pt-4 md:hidden">
              {[
                { label: 'Snippets', Icon: Code2, onClick: onOpenSnippets },
                { label: 'Limits', Icon: SlidersHorizontal, onClick: onOpenConfig },
                { label: 'Pricing', Icon: ReceiptText, onClick: onOpenPricing },
              ].map(({ label, Icon, onClick }) => (
                <button
                  key={label}
                  type="button"
                  onClick={onClick}
                  className="flex h-9 items-center justify-center gap-1.5 rounded-lg border bg-card text-xs font-medium"
                >
                  <Icon className="size-3.5" aria-hidden="true" />
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
