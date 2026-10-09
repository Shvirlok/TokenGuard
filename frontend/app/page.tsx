import { Dashboard } from '@/components/tokenguard/dashboard'
import { TokenGuardProvider } from '@/components/tokenguard/provider'

export default function Page() {
  return (
    <TokenGuardProvider>
      <Dashboard />
    </TokenGuardProvider>
  )
}
