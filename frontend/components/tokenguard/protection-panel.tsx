'use client'

import { Check, Copy, ShieldAlert, Sliders } from 'lucide-react'
import { useState } from 'react'
import { PROFILES } from '@/lib/tokenguard/constants'
import { cn } from '@/lib/utils'
import { useTokenGuard } from './provider'

interface ProtectionPanelProps {
  onOpenConfig?: () => void
}

type TabType = 'python' | 'bash' | 'typescript'

const SNIPPETS: Record<TabType, { label: string; code: string }> = {
  python: {
    label: 'Python',
    code: `from openai import OpenAI

# Route agent traffic through TokenGuard circuit breaker
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="tg-proxy"  # or OPENAI_API_KEY
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Execute agent workflow..."}]
)`,
  },
  bash: {
    label: 'Bash / Env',
    code: `# Route any OpenAI-compatible SDK / CLI:
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"

# Or execute with automatic process wrapper:
tokenguard run python my_agent.py`,
  },
  typescript: {
    label: 'TypeScript',
    code: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
});

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Execute agent workflow..." }],
});`,
  },
}

export function ProtectionPanel({ onOpenConfig }: ProtectionPanelProps) {
  const { stats, setProfile, pending } = useTokenGuard()
  const [activeTab, setActiveTab] = useState<TabType>('python')
  const [copied, setCopied] = useState(false)

  const activeSnippet = SNIPPETS[activeTab]

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeSnippet.code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
    }
  }

  return (
    <section
      aria-label="Agent Config Snippets & Policy"
      className="flex h-full min-h-0 flex-col rounded border border-[#27272a] bg-[#101012] font-mono text-xs"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[#27272a] px-3 py-2">
        <div className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-zinc-200">
          <span className="size-2 rounded-full bg-[#10b981]" />
          <span>Agent Config Snippets</span>
        </div>

        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex h-6 items-center gap-1.5 rounded border border-[#3f3f46] bg-[#18181b] px-2 text-[10px] font-semibold tracking-wider uppercase text-zinc-200 transition-colors hover:border-zinc-500 hover:text-white cursor-pointer select-none"
        >
          {copied ? <Check className="size-2.5 text-[#10b981]" /> : <Copy className="size-2.5" />}
          <span>{copied ? 'COPIED' : 'COPY CODE'}</span>
        </button>
      </div>

      <div className="flex shrink-0 border-b border-[#27272a] bg-[#141416] px-2 text-[11px]">
        {(['python', 'bash', 'typescript'] as TabType[]).map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={cn(
              'border-b-2 px-3 py-1.5 font-medium transition-colors cursor-pointer',
              activeTab === tab
                ? 'border-zinc-200 text-zinc-100 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            )}
          >
            {SNIPPETS[tab].label}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto bg-[#09090b] p-3 text-zinc-300">
        <pre className="text-[11px] leading-relaxed font-mono whitespace-pre select-text">
          <code>{activeSnippet.code}</code>
        </pre>
      </div>

      <div className="shrink-0 border-t border-[#27272a] bg-[#101012] p-2.5">
        <div className="mb-2 flex items-center justify-between text-[10px] uppercase text-zinc-400">
          <span className="flex items-center gap-1">
            <ShieldAlert className="size-3 text-zinc-400" />
            <span>Active Circuit Breaker Profile</span>
          </span>
          {onOpenConfig && (
            <button
              type="button"
              onClick={onOpenConfig}
              className="inline-flex items-center gap-1 text-[10px] text-zinc-400 hover:text-zinc-200 cursor-pointer"
            >
              <Sliders className="size-2.5" />
              <span>EDIT LIMITS</span>
            </button>
          )}
        </div>

        <div className="grid grid-cols-3 gap-1.5">
          {PROFILES.map((p) => {
            const isActive = (stats?.active_profile ?? 'careful') === p.id
            const isLoading = pending === `profile:${p.id}`

            return (
              <button
                key={p.id}
                type="button"
                disabled={isLoading}
                onClick={() => !isActive && setProfile(p.id)}
                className={cn(
                  'flex flex-col items-start rounded border p-1.5 text-left transition-colors cursor-pointer select-none',
                  isActive
                    ? 'border-[#10b981]/70 bg-[#10b981]/10 text-zinc-100'
                    : 'border-[#27272a] bg-[#141416] text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                )}
              >
                <div className="flex w-full items-center justify-between text-[10px] font-bold uppercase">
                  <span>{p.name}</span>
                  {isActive && <span className="size-1.5 rounded-full bg-[#10b981]" />}
                </div>
                <span className="text-[9px] text-zinc-500 mt-0.5 truncate w-full">
                  ${p.hourly_limit}/hr · {p.loop_threshold > 0 ? `${p.loop_threshold}x cut` : 'monitor'}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </section>
  )
}
