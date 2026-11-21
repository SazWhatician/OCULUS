import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import ThemeSwitcher from "@/components/ThemeSwitcher";

export const metadata: Metadata = {
  title: "VeriMeet AI — Apple-Grade Proctor & Speech Sentinel",
  description: "AI-powered interview proctoring platform with Sarvam AI Saaras v2 real-time Speech-to-Text (STT) and Text-to-Speech (TTS) background agent.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body className="antialiased flex flex-col min-h-screen">
        {/* Premium Translucent Apple Header */}
        <header className="sticky top-0 z-50 bg-[var(--apple-surface-translucent)] backdrop-blur-2xl border-b border-[var(--apple-border)] px-6 py-3 transition-colors shadow-sm">
          <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
            {/* Left: Brand Identity */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-8 h-8 rounded-full bg-[var(--apple-accent)] text-white flex items-center justify-center font-bold text-sm shadow-md group-hover:scale-105 transition-transform">
                
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-base tracking-tight text-[var(--apple-text-primary)] flex items-center gap-2">
                  VeriMeet <span className="text-xs font-bold text-[var(--apple-accent)]">Sentinel</span>
                </span>
                <span className="text-[10px] text-[var(--apple-text-muted)] font-semibold hidden sm:inline">Proctoring & Speech AI</span>
              </div>
            </Link>

            {/* Center: Navigation Tabs */}
            <nav className="hidden md:flex items-center gap-1 bg-[var(--apple-bg)] p-1 rounded-full border border-[var(--apple-border)] text-xs font-semibold">
              <Link
                href="/"
                className="px-4 py-1.5 rounded-full text-[var(--apple-text-primary)] hover:bg-[var(--apple-surface)] transition-all"
              >
                Dashboard
              </Link>

              <Link
                href="/reports"
                className="px-4 py-1.5 rounded-full text-[var(--apple-text-primary)] hover:bg-[var(--apple-surface)] transition-all"
              >
                Integrity Reports
              </Link>

              <Link
                href="/sessions/new"
                className="px-4 py-1.5 rounded-full text-[var(--apple-text-primary)] hover:bg-[var(--apple-surface)] transition-all flex items-center gap-1.5"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse"></span>
                Launch Agent Bot
              </Link>
            </nav>

            {/* Right: Sarvam AI Badge, Theme Dropdown & Launch Button */}
            <div className="flex items-center gap-3">
              {/* Sarvam AI Status */}
              <div className="hidden lg:flex apple-badge items-center gap-2 text-[11px] font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse"></span>
                <span>Sarvam AI Saaras v2</span>
              </div>

              {/* Theme Dropdown */}
              <ThemeSwitcher />

              {/* Quick Launch Button */}
              <Link
                href="/sessions/new"
                className="apple-button-primary px-3.5 py-1.5 text-xs font-bold hidden sm:inline-flex items-center gap-1"
              >
                <span>+ Launch Bot</span>
              </Link>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-10">
          {children}
        </main>

        {/* Apple Footer */}
        <footer className="py-8 px-6 border-t border-[var(--apple-border)] text-center text-xs text-[var(--apple-text-secondary)]">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
            <div>© {new Date().getFullYear()} VeriMeet AI Inc. Powered by Sarvam AI Saaras v2 Speech Processing.</div>
            <div className="flex items-center gap-3">
              <span className="apple-badge">English & Hindi ASR</span>
              <span className="apple-badge">Saaras v2 Speech Agent</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
