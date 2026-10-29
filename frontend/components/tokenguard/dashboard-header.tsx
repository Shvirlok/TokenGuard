'use client'

import { ExternalLink, Power, Shield, SlidersHorizontal, Volume2, VolumeX } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'

interface DashboardHeaderProps {
  onOpenConfig?: () => void
}

export function DashboardHeader({ onOpenConfig }: DashboardHeaderProps) {
  const { stats, mode, baseUrl, pending, toggleKillSwitch, soundEnabled, setSoundEnabled } = useTokenGuard()
  const killed = Boolean(stats?.kill_switch_active)

  const socketAddress = baseUrl
    ? baseUrl.replace(/^https?:\/\//, '')
    : '127.0.0.1:8080'

  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-[#27272a] bg-[#101012] px-3 font-mono">
      <div className="flex items-center gap-2.5">
        <div className="flex items-center gap-1.5">
          <div className="flex size-5 items-center justify-center rounded bg-[#18181b] border border-[#27272a] text-zinc-300">
            <Shield className="size-3 text-zinc-300" />
          </div>
          <span className="text-xs font-bold tracking-tight text-zinc-100 uppercase">TokenGuard</span>
          <span className="rounded border border-[#27272a] bg-[#18181b] px-1.5 py-0.5 text-[10px] text-zinc-400">
            v0.2.3
          </span>
        </div>

        {stats?.dry_run && (
          <span className="inline-flex items-center gap-1.5 rounded border border-amber-500/60 bg-amber-950/40 px-2 py-0.5 text-[10px] font-semibold tracking-wider text-amber-300 animate-pulse uppercase">
            <span className="size-1.5 rounded-full bg-amber-400" />
            SHADOW / DRY-RUN ACTIVE
          </span>
        )}

        <div className="h-3.5 w-px bg-[#27272a]" />

        <a
          href="https://github.com/xerj-org/xerj"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded border border-[#27272a] bg-[#121215] px-2 py-0.5 text-[11px] text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-200"
          title="Indexed in XERJ Corpus Hub"
        >
          <span>Indexed in XERJ Corpus Hub</span>
          <ExternalLink className="size-2.5 opacity-60" />
        </a>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-[11px]">
          <span
            className={cn(
              'size-2 rounded-full',
              mode === 'live'
                ? 'bg-[#10b981]'
                : mode === 'demo'
                  ? 'bg-[#10b981]'
                  : 'bg-zinc-600 animate-pulse'
            )}
            aria-hidden="true"
          />
          <span className="tracking-wide text-zinc-300 font-medium">
            {mode === 'connecting' ? 'CONNECTING…' : `LIVE ON ${socketAddress}`}
          </span>
        </div>

        <div className="h-3.5 w-px bg-[#27272a]" />

        <button
          type="button"
          onClick={toggleKillSwitch}
          disabled={mode === 'connecting' || pending === 'kill'}
          className={cn(
            'inline-flex h-7 items-center gap-1.5 rounded border px-2.5 text-[11px] font-semibold tracking-wider uppercase transition-all disabled:opacity-50 select-none cursor-pointer',
            killed
              ? 'border-rose-500 bg-rose-950/40 text-rose-400 hover:bg-rose-900/50'
              : 'border-[#3f3f46] bg-[#18181b] text-zinc-300 hover:border-rose-700 hover:bg-rose-950/20 hover:text-rose-400'
          )}
        >
          <Power className={cn('size-3', killed ? 'text-rose-400 animate-pulse' : 'text-zinc-500')} />
          <span>{killed ? 'KILL SWITCH: ENGAGED' : 'KILL SWITCH'}</span>
        </button>

        <div className="flex items-center gap-1">
          {onOpenConfig && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <button
                    type="button"
                    onClick={onOpenConfig}
                    aria-label="Guard Limits"
                    className="flex size-7 items-center justify-center rounded border border-[#27272a] bg-[#141416] text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 cursor-pointer"
                  />
                }
              >
                <SlidersHorizontal className="size-3" />
              </TooltipTrigger>
              <TooltipContent className="font-mono text-[10px] bg-[#101012] border border-[#27272a] text-zinc-300">
                Guard Limits
              </TooltipContent>
            </Tooltip>
          )}

          <Tooltip>
            <TooltipTrigger
              render={
                <button
                  type="button"
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  aria-label={soundEnabled ? 'Mute Alerts' : 'Unmute Alerts'}
                  className={cn(
                    'flex size-7 items-center justify-center rounded border transition-colors cursor-pointer',
                    soundEnabled
                      ? 'border-[#10b981]/50 bg-[#10b981]/10 text-[#10b981]'
                      : 'border-[#27272a] bg-[#141416] text-zinc-500 hover:text-zinc-300'
                  )}
                />
              }
            >
              {soundEnabled ? <Volume2 className="size-3" /> : <VolumeX className="size-3" />}
            </TooltipTrigger>
            <TooltipContent className="font-mono text-[10px] bg-[#101012] border border-[#27272a] text-zinc-300">
              {soundEnabled ? 'Audio Alerts: ON' : 'Audio Alerts: MUTED'}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </header>
  )
}
