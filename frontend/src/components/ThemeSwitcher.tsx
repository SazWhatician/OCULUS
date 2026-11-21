"use client";

import { useState, useEffect, useRef } from "react";

export type ThemeOption = "light" | "dark" | "cream" | "warm" | "sage" | "obsidian" | "ocean";

interface ThemeItem {
  id: ThemeOption;
  name: string;
  category: "Light" | "Dark";
  colorPreview: string; // Tailwind gradient / swatch
  icon: string;
}

const THEMES: ThemeItem[] = [
  { id: "light", name: "Apple Light", category: "Light", colorPreview: "bg-slate-200 border-blue-500", icon: "☀️" },
  { id: "cream", name: "Vanilla Cream", category: "Light", colorPreview: "bg-amber-100 border-amber-600", icon: "🍦" },
  { id: "warm", name: "Warm Sandstone", category: "Light", colorPreview: "bg-orange-200 border-orange-700", icon: "🏜️" },
  { id: "sage", name: "Nordic Sage", category: "Light", colorPreview: "bg-emerald-100 border-emerald-600", icon: "🌲" },
  { id: "dark", name: "Apple Dark", category: "Dark", colorPreview: "bg-black border-blue-400", icon: "🌙" },
  { id: "obsidian", name: "Velvet Obsidian", category: "Dark", colorPreview: "bg-purple-950 border-purple-500", icon: "🍇" },
  { id: "ocean", name: "Oceanic Azure", category: "Dark", colorPreview: "bg-sky-950 border-sky-400", icon: "🌊" },
];

export default function ThemeSwitcher() {
  const [currentTheme, setCurrentTheme] = useState<ThemeOption>("light");
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    const savedTheme = (localStorage.getItem("verimeet_theme") as ThemeOption) || "light";
    const validTheme = THEMES.some((t) => t.id === savedTheme) ? savedTheme : "light";
    setCurrentTheme(validTheme);
    document.documentElement.setAttribute("data-theme", validTheme);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectTheme = (theme: ThemeOption) => {
    setCurrentTheme(theme);
    localStorage.setItem("verimeet_theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
    setIsOpen(false);
  };

  if (!mounted) return null;

  const activeThemeObj = THEMES.find((t) => t.id === currentTheme) || THEMES[0];

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="apple-button-secondary px-3.5 py-1.5 text-xs flex items-center gap-2 font-semibold transition-all shadow-sm"
      >
        <span>{activeThemeObj.icon}</span>
        <span className="hidden sm:inline">{activeThemeObj.name}</span>
        <svg
          className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 rounded-2xl bg-[var(--apple-surface-translucent)] backdrop-blur-2xl border border-[var(--apple-border)] shadow-2xl p-1.5 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
          <div className="px-3 py-1.5 text-[10px] font-extrabold uppercase tracking-wider text-[var(--apple-text-muted)] border-b border-[var(--apple-border)] mb-1">
            Theme Palette ({THEMES.length} Options)
          </div>

          <div className="space-y-0.5">
            {THEMES.map((t) => {
              const isSelected = currentTheme === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => selectTheme(t.id)}
                  className={`w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center justify-between transition-all ${
                    isSelected
                      ? "bg-[var(--apple-accent)] text-white shadow-sm font-bold"
                      : "text-[var(--apple-text-primary)] hover:bg-[var(--apple-bg)]"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-sm">{t.icon}</span>
                    <span>{t.name}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full border ${t.colorPreview}`}></span>
                    {isSelected && (
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
