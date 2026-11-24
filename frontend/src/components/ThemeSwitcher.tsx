"use client";

import React, { useState, useEffect, useRef } from "react";

export type ThemeOption = "light" | "dark" | "cream" | "warm" | "sage" | "obsidian" | "ocean";

interface ThemeItem {
  id: ThemeOption;
  name: string;
  polarity: "LIGHT" | "DARK";
  accent: string;
}

const THEMES: ThemeItem[] = [
  { id: "light", name: "Paper Bone", polarity: "LIGHT", accent: "#0A0A0A" },
  { id: "dark", name: "Pitch Charcoal", polarity: "DARK", accent: "#FBFBF9" },
  { id: "cream", name: "Vanilla Cream", polarity: "LIGHT", accent: "#B45309" },
  { id: "warm", name: "Warm Sandstone", polarity: "LIGHT", accent: "#C2410C" },
  { id: "sage", name: "Nordic Sage", polarity: "LIGHT", accent: "#059669" },
  { id: "obsidian", name: "Velvet Obsidian", polarity: "DARK", accent: "#A855F7" },
  { id: "ocean", name: "Deep Ocean", polarity: "DARK", accent: "#38BDF8" },
];

export default function ThemeSwitcher() {
  const [currentTheme, setCurrentTheme] = useState<ThemeOption>("light");
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const isDark = currentTheme === "dark" || currentTheme === "obsidian" || currentTheme === "ocean";

  useEffect(() => {
    setMounted(true);
    const saved =
      (localStorage.getItem("oculus_theme") as ThemeOption) ||
      (localStorage.getItem("verimeet_theme") as ThemeOption) ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    const valid = THEMES.some((t) => t.id === saved) ? saved : "light";
    applyTheme(valid, false);
  }, []);

  const applyTheme = (theme: ThemeOption, dispatch = true) => {
    setCurrentTheme(theme);
    localStorage.setItem("oculus_theme", theme);
    localStorage.setItem("verimeet_theme", theme);

    const doc = document.documentElement;
    doc.setAttribute("data-theme", theme);
    const darkNow = theme === "dark" || theme === "obsidian" || theme === "ocean";
    if (darkNow) {
      doc.classList.add("dark");
    } else {
      doc.classList.remove("dark");
    }

    if (dispatch && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("oculus-theme-change", { detail: { theme, darkNow } }));
    }
  };

  // 1-Click Polarity Inverter: Invert between pure Light (#FBFBF9) and pure Dark (#0A0A0A)
  const togglePolarity = () => {
    const nextTheme: ThemeOption = isDark ? "light" : "dark";
    applyTheme(nextTheme);
  };

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!mounted) {
    return (
      <div className="h-7 w-24 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] opacity-40 animate-pulse" />
    );
  }

  return (
    <div className="relative inline-flex items-center" ref={containerRef}>
      {/* Primary Inverter Button */}
      <button
        type="button"
        onClick={togglePolarity}
        title={isDark ? "Invert color to Light (black becomes white)" : "Invert color to Dark (white becomes black)"}
        aria-label="Invert color polarity"
        className="group relative flex items-center gap-2 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] px-2.5 py-1.5 font-mono text-[9.5px] uppercase tracking-[0.24em] text-[var(--oculus-ink)] transition-all hover:border-[var(--oculus-ink)] hover:bg-[var(--oculus-bg)] focus:outline-none"
      >
        {/* Optical Aperture Half-Circle Inverter Glyphs */}
        <span className="relative flex h-3.5 w-3.5 items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:scale-110">
          <svg
            viewBox="0 0 16 16"
            className={`h-3.5 w-3.5 transition-transform duration-700 ease-[cubic-bezier(0.19,1,0.22,1)] ${isDark ? "rotate-180" : "rotate-0"
              }`}
            fill="currentColor"
            aria-hidden="true"
          >
            {/* Outer border circle */}
            <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.2" />
            {/* Filled half for optical contrast */}
            <path d="M 8 1 A 7 7 0 0 1 8 15 Z" fill="currentColor" />
          </svg>
        </span>

        {/* Readout */}
        <span className="font-semibold tracking-[0.26em]">
          INV // <span className="tabular-nums">{isDark ? "LIGHT" : "DARK"}</span>
        </span>
      </button>

      {/* Palette Selector Trigger */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        title="More tonal calibrations"
        aria-label="Open tonal palette options"
        className="flex items-center justify-center border-y border-r border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] px-1.5 py-1.5 font-mono text-[9px] text-[var(--oculus-muted)] transition-colors hover:border-[var(--oculus-ink)] hover:text-[var(--oculus-ink)]"
      >
        <span className={`transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`}>▾</span>
      </button>

      {/* Dropdown for other tonal calibrated palettes */}
      {isOpen && (
        <div className="absolute right-0 top-full z-50 mt-1.5 w-48 border border-[var(--oculus-hairline)] bg-[var(--oculus-surface)] p-1 shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-top-1 duration-150 font-mono text-[10px]">
          <div className="border-b border-[var(--oculus-hairline)] px-2.5 py-1.5 text-[8.5px] uppercase tracking-[0.28em] text-[var(--oculus-muted)]">
            OPTICAL CALIBRATIONS
          </div>

          <div className="mt-1 flex flex-col gap-0.5">
            {THEMES.map((t) => {
              const active = currentTheme === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => {
                    applyTheme(t.id);
                    setIsOpen(false);
                  }}
                  className={`flex items-center justify-between px-2.5 py-1.5 text-left transition-colors ${active
                      ? "bg-[var(--oculus-ink)] text-[var(--oculus-bg)] font-semibold"
                      : "text-[var(--oculus-ink)] hover:bg-[var(--oculus-bg)]"
                    }`}
                >
                  <span className="tracking-[0.14em] uppercase">{t.name}</span>
                  <span className="text-[8px] uppercase tracking-[0.2em] opacity-60">
                    {t.polarity}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}


