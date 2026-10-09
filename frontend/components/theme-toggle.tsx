'use client'

import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useSyncExternalStore } from 'react'
import { cn } from '@/lib/utils'

const subscribe = () => () => {}

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const mounted = useSyncExternalStore(subscribe, () => true, () => false)
  const active = mounted ? resolvedTheme : 'dark'

  const options = [
    { value: 'dark', label: 'Dark mode', Icon: Moon },
    { value: 'light', label: 'Light mode', Icon: Sun },
  ] as const

  return (
    <div role="radiogroup" aria-label="Theme" className="flex items-center rounded-full border bg-card p-0.5">
      {options.map(({ value, label, Icon }) => {
        const selected = active === value
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={label}
            onClick={() => setTheme(value)}
            className={cn(
              'flex size-7 items-center justify-center rounded-full transition-colors',
              selected ? 'bg-secondary text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            <Icon className="size-3.5" aria-hidden="true" />
          </button>
        )
      })}
    </div>
  )
}
