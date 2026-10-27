import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import './globals.css'

const geistSans = Geist({ subsets: ['latin'], variable: '--font-geist-sans' })
const geistMono = Geist_Mono({ subsets: ['latin'], variable: '--font-geist-mono' })

export const metadata: Metadata = {
  title: 'TokenGuard — LLM Circuit Breaker HUD',
  description:
    'Fail-closed local proxy & circuit breaker for AI agents: loop interception, budget preservation, and real-time developer stream.',
}

export const viewport: Viewport = {
  colorScheme: 'dark',
  themeColor: '#09090b',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`dark h-screen overflow-hidden ${geistSans.variable} ${geistMono.variable}`}>
      <body className="font-sans antialiased bg-[#09090b] text-[#e4e4e7] h-screen max-h-screen overflow-hidden select-none">
        <ThemeProvider attribute="class" defaultTheme="dark" forcedTheme="dark" enableSystem={false} disableTransitionOnChange>
          <TooltipProvider>{children}</TooltipProvider>
          <Toaster
            position="bottom-right"
            theme="dark"
            toastOptions={{
              className: 'font-mono text-xs border border-[#27272a] bg-[#101012] text-zinc-100',
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  )
}
