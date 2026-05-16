import { Link, useLocation } from 'react-router-dom'
import { useState } from 'react'

const navLinks = [
  { to: '/', label: 'Leaderboard' },
  { to: '/compare', label: 'Compare' },
  { to: '/about', label: 'About' },
]

export default function Header() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 bg-dm-bg/80 backdrop-blur-lg border-b border-dm-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-8 h-8 rounded-lg bg-dm-accent flex items-center justify-center group-hover:glow-accent transition-shadow">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <div className="hidden sm:block">
              <span className="text-white font-semibold text-lg tracking-tight">StabilityBench</span>
              <span className="text-dm-muted text-xs block leading-none mt-0.5">RSI Leaderboard</span>
            </div>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  location.pathname === link.to
                    ? 'bg-dm-accent/10 text-dm-accent-light'
                    : 'text-dm-muted hover:text-white hover:bg-dm-surface'
                }`}
              >
                {link.label}
              </Link>
            ))}
            <a
              href="/api"
              className="px-4 py-2 rounded-lg text-sm font-medium text-dm-muted hover:text-white hover:bg-dm-surface transition-all duration-200"
            >
              API
            </a>
          </nav>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg text-dm-muted hover:text-white hover:bg-dm-surface transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {mobileOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Nav */}
        {mobileOpen && (
          <nav className="md:hidden pb-4 border-t border-dm-border mt-2 pt-3 space-y-1">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMobileOpen(false)}
                className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  location.pathname === link.to
                    ? 'bg-dm-accent/10 text-dm-accent-light'
                    : 'text-dm-muted hover:text-white hover:bg-dm-surface'
                }`}
              >
                {link.label}
              </Link>
            ))}
            <a
              href="/api"
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-2.5 rounded-lg text-sm font-medium text-dm-muted hover:text-white hover:bg-dm-surface transition-all"
            >
              API
            </a>
          </nav>
        )}
      </div>
    </header>
  )
}
