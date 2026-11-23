import type { Metadata } from "next";
import "./globals.css";
import OculusNavbar from "@/components/OculusNavbar";
import OculusLoader from "@/components/OculusLoader";

export const metadata: Metadata = {
  title: "OCULUS Sentinel — AI Interview Integrity Platform",
  description:
    "Event-Driven Multimodal Behavioral Sentinel with Hybrid STT, OpenCV Vision Telemetry, and Q0 Baseline-Calibrated LLM Judge.",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body className="min-h-screen flex flex-col bg-[var(--oculus-bg)] text-[var(--oculus-ink)] font-sans antialiased selection:bg-[var(--oculus-ink)] selection:text-[var(--oculus-bg)]">
        <OculusLoader oncePerSession={false} />
        <OculusNavbar />
        <main className="w-full flex-1 pt-[74px]">{children}</main>
        <footer className="border-t border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] px-6 py-10 font-mono text-xs uppercase tracking-[0.2em] text-[var(--oculus-muted)]">
          <div className="mx-auto flex max-w-[1380px] flex-col items-center justify-between gap-6 sm:flex-row">
            <div className="flex items-center gap-3">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>OCULUS SENTINEL // CORE ACTIVE</span>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-6 text-[10px]">
              <span>SPEECHMATICS & GROQ WHISPER</span>
              <span>OPENCV & GEMINI VLM</span>
              <span>Q0 MEDIAN & MAD CALIBRATION</span>
            </div>
            <div>© {new Date().getFullYear()} // ADVISORY PROCTORING ONLY</div>
          </div>
        </footer>
      </body>
    </html>
  );
}
