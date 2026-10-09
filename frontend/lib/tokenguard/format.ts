export function parseTimestamp(value: string): number {
  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  const t = Date.parse(normalized)
  return Number.isNaN(t) ? Date.now() : t
}

export function formatUsd(value: number, digits = 4) {
  return `$${(value || 0).toFixed(digits)}`
}

export function formatSaved(value: number) {
  const v = Number(value || 0)
  return `+$${v > 0 && v < 0.01 ? v.toFixed(4) : v.toFixed(2)}`
}

export function formatNumber(value: number) {
  return (value || 0).toLocaleString('en-US')
}

export function formatTime(value: string) {
  return new Date(parseTimestamp(value)).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function formatDateTime(value: string) {
  return new Date(parseTimestamp(value)).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function isBlocked(status: string) {
  return status.startsWith('blocked')
}
