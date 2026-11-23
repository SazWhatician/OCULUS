import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import ThemeSwitcher from "@/components/ThemeSwitcher";

export const metadata: Metadata = {
  title: "OCULUS Sentinel — AI Interview Integrity & Proctoring Platform",
  description:
    "Event-Driven Multimodal Interview Integrity Sentinel with Hybrid STT, OpenCV Vision Telemetry, and Q0 Baseline-Calibrated LLM Judge.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body className="antialiased flex flex-col min-h-screen">
        {/* Apple Translucent Glass Header */}
        <header className="sticky top-0 z-50 bg-[var(--apple-surface-translucent)] backdrop-blur-2xl border-b border-[var(--apple-border)] px-6 py-3 transition-colors shadow-sm">
          <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
            {/* Left: Brand Identity */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-2xl bg-gradient-to-tr from-[var(--apple-accent)] to-sky-400 text-white flex items-center justify-center font-black text-base shadow-md group-hover:scale-105 transition-transform">
                👁
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-base tracking-tight text-[var(--apple-text-primary)] flex items-center gap-2">
                  OCULUS <span className="text-xs font-bold text-[var(--apple-accent)]">Sentinel</span>
                </span>
                <span className="text-[10px] text-[var(--apple-text-muted)] font-semibold hidden sm:inline">
                  AI Interview Integrity Sentinel
                </span>
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
                Audit Reports
              </Link>

              <Link
                href="/sessions/new"
                className="px-4 py-1.5 rounded-full text-[var(--apple-text-primary)] hover:bg-[var(--apple-surface)] transition-all flex items-center gap-1.5"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse"></span>
                Launch Sentinel
              </Link>
            </nav>

            {/* Right: Engine Badges & Quick Launch */}
            <div className="flex items-center gap-3">
              <div className="hidden lg:flex apple-badge items-center gap-2 text-[11px] font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse"></span>
                <span>Hybrid STT • Llama 3.3</span>
              </div>

              <ThemeSwitcher />

              <Link
                href="/sessions/new"
                className="apple-button-primary px-4 py-1.5 text-xs font-bold hidden sm:inline-flex items-center gap-1.5"
              >
                <span>+ Launch Sentinel</span>
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
            <div>
              © {new Date().getFullYear()} <strong>OCULUS Sentinel</strong>. Advisory Behavioral Interview Proctoring.
            </div>
            <div className="flex items-center gap-2 flex-wrap justify-center">
              <span className="apple-badge">Speechmatics & Groq Whisper</span>
              <span className="apple-badge">OpenCV & Gemini VLM</span>
              <span className="apple-badge">Q0 Baseline Calibrated</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
