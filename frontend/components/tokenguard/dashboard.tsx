'use client'

import { FlaskConical, Loader2, RefreshCw, Wifi } from 'lucide-react'
import { useState } from 'react'
import type { RequestRecord } from '@/lib/tokenguard/types'
import { DEFAULT_PROXY_URL } from '@/lib/tokenguard/constants'
import { DashboardHeader } from './dashboard-header'
import { ClearLogsDialog, ConfigDialog, InspectorDialog, PricingDialog, SnippetsDialog } from './dialogs'
import { MetricCards } from './metric-cards'
import { ProtectionPanel } from './protection-panel'
import { useTokenGuard } from './provider'
import { RequestsTable } from './requests-table'
import { SpendChart } from './spend-chart'

function ConnectionBar() {
  const { mode, baseUrl, retryConnection } = useTokenGuard()

  if (mode === 'connecting') {
    return (
      <div className="flex items-center gap-2 rounded-xl border bg-card px-4 py-2.5 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
        Looking for TokenGuard proxy…
      </div>
    )
  }

  if (mode === 'live') {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-success/30 bg-success/5 px-4 py-2.5 text-xs text-success">
        <Wifi className="size-3.5" aria-hidden="true" />
        Connected to proxy at <code className="font-mono">{baseUrl || 'this origin'}</code>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-warning/30 bg-warning/5 px-4 py-2.5 text-xs sm:flex-row sm:items-center sm:justify-between">
      <p className="flex items-start gap-2 text-warning sm:items-center">
        <FlaskConical className="mt-0.5 size-3.5 shrink-0 sm:mt-0" aria-hidden="true" />
        <span>
          Demo mode — no proxy found at <code className="font-mono">{DEFAULT_PROXY_URL}</code>. Every control works
          against an in-browser simulation. Start it with <code className="font-mono">tokenguard start</code> to see your
          real traffic.
        </span>
      </p>
      <button
        type="button"
        onClick={retryConnection}
        className="inline-flex h-7 shrink-0 items-center gap-1.5 self-start rounded-lg border border-warning/40 px-2.5 font-medium text-warning hover:bg-warning/10 sm:self-auto"
      >
        <RefreshCw className="size-3" aria-hidden="true" />
        Retry
      </button>
    </div>
  )
}

export function Dashboard() {
  const [snippetsOpen, setSnippetsOpen] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [pricingOpen, setPricingOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)
  const [inspected, setInspected] = useState<RequestRecord | null>(null)

  const openers = {
    onOpenSnippets: () => setSnippetsOpen(true),
    onOpenConfig: () => setConfigOpen(true),
    onOpenPricing: () => setPricingOpen(true),
  }

  return (
    <div className="min-h-dvh">
      <DashboardHeader {...openers} />
      <main className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 sm:py-8">
        <h1 className="sr-only">TokenGuard dashboard</h1>
        <ConnectionBar />
        <ProtectionPanel {...openers} />
        <MetricCards />
        <SpendChart />
        <RequestsTable onInspect={setInspected} onClearLogs={() => setClearOpen(true)} />
      </main>

      <SnippetsDialog open={snippetsOpen} onOpenChange={setSnippetsOpen} />
      <ConfigDialog open={configOpen} onOpenChange={setConfigOpen} />
      <PricingDialog open={pricingOpen} onOpenChange={setPricingOpen} />
      <ClearLogsDialog open={clearOpen} onOpenChange={setClearOpen} />
      <InspectorDialog record={inspected} onClose={() => setInspected(null)} />
    </div>
  )
}
