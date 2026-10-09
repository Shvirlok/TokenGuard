'use client'

import { Ban, Code2, Power, ReceiptText, ShieldCheck, SlidersHorizontal, Volume2, VolumeX, Zap } from 'lucide-react'
import { ThemeToggle } from '@/components/theme-toggle'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'

interface DashboardHeaderProps {
  onOpenSnippets: () => void
  onOpenConfig: () => void
  onOpenPricing: () => void
}

function IconButton({
  label,
  onClick,
  children,
  active,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
  active?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            onClick={onClick}
            aria-label={label}
            aria-pressed={active}
            className={cn(
              'flex size-9 items-center justify-center rounded-xl border bg-card text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground',
              active && 'border-primary/50 text-primary',
            )}
          />
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

export function DashboardHeader({ onOpenSnippets, onOpenConfig, onOpenPricing }: DashboardHeaderProps) {
  const { stats, polling, mode, pending, simulate, toggleKillSwitch, soundEnabled, setSoundEnabled } = useTokenGuard()
  const killed = Boolean(stats?.kill_switch_active)
  const state = killed ? 'killed' : !polling ? 'paused' : 'guarding'

  const pill = {
    guarding: { label: 'Guarding', className: 'border-success/40 bg-success/10 text-success', dot: 'bg-success' },
    paused: { label: 'Paused', className: 'border-warning/40 bg-warning/10 text-warning', dot: 'bg-warning' },
    killed: {
      label: 'Killed',
      className: 'border-destructive/50 bg-destructive/10 text-destructive shadow-[0_0_24px_-6px] shadow-destructive/60',
      dot: 'bg-destructive',
    },
  }[state]

  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-lg shadow-primary/30">
            <ShieldCheck className="size-5" aria-hidden="true" />
          </div>
          <span className="hidden text-sm font-semibold tracking-tight md:inline">TokenGuard</span>
        </div>

        <div
          role="status"
          aria-live="polite"
          className={cn(
            'flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-bold uppercase tracking-wider transition-colors',
            pill.className,
          )}
        >
          <span className="relative flex size-2">
            {state !== 'paused' && (
              <span className={cn('absolute inline-flex size-full animate-ping rounded-full opacity-70', pill.dot)} />
            )}
            <span className={cn('relative inline-flex size-2 rounded-full', pill.dot)} />
          </span>
          {pill.label}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => simulate('loop')}
            disabled={mode === 'connecting' || pending === 'sim:loop'}
            className="hidden h-9 items-center gap-1.5 rounded-xl border border-info/40 bg-info/10 px-3 text-xs font-semibold text-info transition-colors hover:bg-info/20 disabled:opacity-50 sm:flex"
          >
            <Zap className="size-3.5" aria-hidden="true" />
            Test Breaker
          </button>
          <button
            type="button"
            onClick={toggleKillSwitch}
            disabled={mode === 'connecting' || pending === 'kill'}
            className={cn(
              'flex h-9 items-center gap-1.5 rounded-xl border px-3 text-xs font-semibold transition-colors disabled:opacity-50',
              killed
                ? 'border-success/40 bg-success/10 text-success hover:bg-success/20'
                : 'border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/20',
            )}
          >
            {killed ? <Power className="size-3.5" aria-hidden="true" /> : <Ban className="size-3.5" aria-hidden="true" />}
            <span className="hidden sm:inline">{killed ? 'Resume All Requests' : 'Kill All Requests'}</span>
            <span className="sm:hidden">{killed ? 'Resume' : 'Kill'}</span>
          </button>

          <div className="mx-1 hidden h-6 w-px bg-border sm:block" aria-hidden="true" />

          <div className="hidden items-center gap-2 md:flex">
            <IconButton label="Integration snippets" onClick={onOpenSnippets}>
              <Code2 className="size-4" aria-hidden="true" />
            </IconButton>
            <IconButton label="Guard limits" onClick={onOpenConfig}>
              <SlidersHorizontal className="size-4" aria-hidden="true" />
            </IconButton>
            <IconButton label="Model pricing" onClick={onOpenPricing}>
              <ReceiptText className="size-4" aria-hidden="true" />
            </IconButton>
          </div>
          <IconButton
            label={soundEnabled ? 'Mute block alerts' : 'Enable block alerts'}
            onClick={() => setSoundEnabled(!soundEnabled)}
            active={soundEnabled}
          >
            {soundEnabled ? (
              <Volume2 className="size-4" aria-hidden="true" />
            ) : (
              <VolumeX className="size-4" aria-hidden="true" />
            )}
          </IconButton>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
