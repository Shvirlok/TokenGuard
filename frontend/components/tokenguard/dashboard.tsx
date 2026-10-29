'use client'

import { useState } from 'react'
import type { RequestRecord } from '@/lib/tokenguard/types'
import { DashboardHeader } from './dashboard-header'
import { ClearLogsDialog, ConfigDialog, InspectorDialog } from './dialogs'
import { MetricCards } from './metric-cards'
import { ProtectionPanel } from './protection-panel'
import { RequestsTable } from './requests-table'

export function Dashboard() {
  const [configOpen, setConfigOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)
  const [inspected, setInspected] = useState<RequestRecord | null>(null)

  return (
    <div className="flex h-screen max-h-screen w-screen flex-col overflow-hidden bg-[#09090b] text-[#e4e4e7] p-2.5 gap-2 select-none font-sans">
      <h1 className="sr-only">TokenGuard — Developer Circuit Breaker HUD</h1>

      <DashboardHeader onOpenConfig={() => setConfigOpen(true)} />

      <MetricCards />

      <div className="flex-1 min-h-0 grid grid-cols-12 gap-2">
        <div className="col-span-12 lg:col-span-7 xl:col-span-8 min-h-0 h-full">
          <RequestsTable
            onInspect={setInspected}
            onClearLogs={() => setClearOpen(true)}
          />
        </div>

        <div className="col-span-12 lg:col-span-5 xl:col-span-4 min-h-0 h-full">
          <ProtectionPanel onOpenConfig={() => setConfigOpen(true)} />
        </div>
      </div>

      <footer className="flex h-5 shrink-0 items-center justify-between px-2 font-mono text-[11px] text-zinc-500 border-t border-[#27272a]/60">
        <div>Architecture: Fail-closed local proxy • Zero-telemetry • MIT License</div>
        <div className="hidden sm:flex items-center gap-2 text-zinc-600">
          <span>127.0.0.1:8080</span>
          <span>•</span>
          <span className="text-[#10b981]">SYSTEM OPERATIONAL</span>
        </div>
      </footer>

      <ConfigDialog open={configOpen} onOpenChange={setConfigOpen} />
      <ClearLogsDialog open={clearOpen} onOpenChange={setClearOpen} />
      <InspectorDialog record={inspected} onClose={() => setInspected(null)} />
    </div>
  )
}
