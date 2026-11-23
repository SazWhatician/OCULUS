"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeSwitcher from "@/components/ThemeSwitcher";
import OculusLogo from "@/components/OculusLogo";

const NAV_ITEMS = [
  { href: "/", index: "01", label: "DASHBOARD" },
  { href: "/reports", index: "02", label: "AUDIT ARCHIVE" },
  { href: "/sessions/new", index: "03", label: "DISPATCH" },
];

function MicroOculusMark() {
  const pupilRef = useRef<SVGCircleElement>(null);
  const stateRef = useRef({ x: 0, y: 0, vx: 0, vy: 0, tx: 0, ty: 0 });

  useEffect(() => {
    let rafId = 0;

    const onMove = (e: MouseEvent) => {
      const nx = (e.clientX / window.innerWidth - 0.15) * 2;
      const ny = (e.clientY / window.innerHeight - 0.1) * 2;
      const angle = Math.atan2(ny, nx);
      const mag = Math.min(Math.hypot(nx, ny), 1);
      stateRef.current.tx = Math.cos(angle) * 24 * mag;
      stateRef.current.ty = Math.sin(angle) * 12 * mag;
    };

    window.addEventListener("mousemove", onMove, { passive: true });

    const tick = () => {
      const s = stateRef.current;
      s.vx = (s.vx + (s.tx - s.x) * 0.1) * 0.76;
      s.vy = (s.vy + (s.ty - s.y) * 0.1) * 0.76;
      s.x += s.vx;
      s.y += s.vy;

      if (pupilRef.current) {
        pupilRef.current.setAttribute("cx", (240 + s.x).toFixed(2));
        pupilRef.current.setAttribute("cy", (180 + s.y).toFixed(2));
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("mousemove", onMove);
    };
  }, []);

  return (
    <OculusLogo
      className="h-7 w-10 text-[var(--oculus-ink)] transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:scale-105"
      pupilRef={pupilRef}
    />
  );
}

function MagneticLink({
  href,
  children,
  className = "",
}: {
  href: string;
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLAnchorElement>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const onMove = (e: React.MouseEvent<HTMLAnchorElement>) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = (e.clientX - (rect.left + rect.width / 2)) * 0.22;
    const y = (e.clientY - (rect.top + rect.height / 2)) * 0.28;
    setOffset({ x, y });
  };

  const onLeave = () => setOffset({ x: 0, y: 0 });

  return (
    <Link
      ref={ref}
      href={href}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{
        transform: `translate3d(${offset.x}px, ${offset.y}px, 0)`,
      }}
      className={`transition-transform duration-300 ease-[cubic-bezier(0.19,1,0.22,1)] ${className}`}
    >
      {children}
    </Link>
  );
}

export default function OculusNavbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [progress, setProgress] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [reticle, setReticle] = useState({ left: 0, width: 0, visible: false });

  const navRef = useRef<HTMLElement>(null);
  const itemRefs = useRef<(HTMLAnchorElement | null)[]>([]);

  const activeIdx = NAV_ITEMS.findIndex((item) =>
    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
  );
  const targetIdx = hoveredIdx !== null ? hoveredIdx : activeIdx;

  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY;
      setScrolled(y > 20);
      const max =
        document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(Math.max(y / max, 0), 1) : 0);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const updateReticle = () => {
      const container = navRef.current;
      const el = targetIdx >= 0 ? itemRefs.current[targetIdx] : null;
      if (!container || !el) {
        setReticle((r) => ({ ...r, visible: false }));
        return;
      }
      const cRect = container.getBoundingClientRect();
      const eRect = el.getBoundingClientRect();
      setReticle({
        left: eRect.left - cRect.left,
        width: eRect.width,
        visible: true,
      });
    };

    updateReticle();
    window.addEventListener("resize", updateReticle);
    return () => window.removeEventListener("resize", updateReticle);
  }, [targetIdx, pathname]);

  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  return (
    <>
      <header
        className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] ${
          scrolled ? "py-3" : "py-5"
        }`}
      >
        <div
          aria-hidden="true"
          className={`pointer-events-none absolute inset-x-0 -bottom-8 top-0 transition-opacity duration-500 ${
            scrolled ? "opacity-100" : "opacity-0"
          }`}
          style={{
            background:
              "linear-gradient(to bottom, var(--oculus-bg) 0%, color-mix(in srgb, var(--oculus-bg) 86%, transparent) 62%, transparent 100%)",
            backdropFilter: "blur(14px)",
            WebkitBackdropFilter: "blur(14px)",
            maskImage:
              "linear-gradient(to bottom, black 64%, transparent 100%)",
            WebkitMaskImage:
              "linear-gradient(to bottom, black 64%, transparent 100%)",
          }}
        />

        <div
          aria-hidden="true"
          className="pointer-events-none fixed left-0 right-0 top-0 z-50 h-[1.5px] origin-left bg-[var(--oculus-ink)] transition-transform duration-150 ease-out"
          style={{ transform: `scaleX(${progress})` }}
        />

        <div className="relative z-10 mx-auto flex max-w-[1380px] items-center justify-between px-5 sm:px-8">
          <Link
            href="/"
            className="group flex items-center gap-3.5 focus:outline-none"
          >
            <MicroOculusMark />
            <div className="flex flex-col leading-none">
              <div className="flex items-baseline gap-2">
                <span className="font-display text-[17px] font-semibold tracking-[0.06em] text-[var(--oculus-ink)]">
                  OCULUS
                </span>
                <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
                  SENTINEL
                </span>
              </div>
              <span className="mt-1 hidden font-mono text-[8.5px] uppercase tracking-[0.28em] text-[var(--oculus-muted)] sm:block">
                Y:{String(Math.round(progress * 100)).padStart(2, "0")}% //{" "}
                {pathname === "/"
                  ? "INDEX"
                  : pathname.replace("/", "").toUpperCase()}
              </span>
            </div>
          </Link>

          <nav
            ref={navRef}
            aria-label="Primary Navigation"
            onMouseLeave={() => setHoveredIdx(null)}
            className="relative hidden items-center gap-9 md:flex"
          >
            <span
              aria-hidden="true"
              style={{
                transform: `translate3d(${reticle.left - 10}px, 0, 0)`,
                width: `${reticle.width + 20}px`,
                opacity: reticle.visible ? 1 : 0,
              }}
              className="pointer-events-none absolute -inset-y-1.5 flex items-center justify-between transition-all duration-500 ease-[cubic-bezier(0.19,1,0.22,1)]"
            >
              <span className="h-3 w-[1.5px] bg-[var(--oculus-ink)]/55" />
              <span className="absolute bottom-0 left-2.5 right-2.5 h-px bg-[var(--oculus-ink)]" />
              <span className="h-3 w-[1.5px] bg-[var(--oculus-ink)]/55" />
            </span>

            {NAV_ITEMS.map((item, idx) => {
              const isActive = idx === activeIdx;
              const isDispatch = item.href === "/sessions/new";
              return (
                <Link
                  key={item.href}
                  ref={(el) => {
                    itemRefs.current[idx] = el;
                  }}
                  href={item.href}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  className="group relative flex items-center gap-2 py-1 font-mono text-[10.5px] uppercase tracking-[0.26em] text-[var(--oculus-ink)] focus:outline-none"
                >
                  <span
                    className={`text-[9px] transition-opacity duration-300 ${
                      isActive
                        ? "opacity-100"
                        : "opacity-35 group-hover:opacity-80"
                    }`}
                  >
                    {item.index}
                  </span>

                  <span className="relative block h-[14px] overflow-hidden leading-[14px]">
                    <span className="block transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:-translate-y-full">
                      {item.label}
                    </span>
                    <span
                      aria-hidden="true"
                      className="block transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:-translate-y-full"
                    >
                      {item.label}
                    </span>
                  </span>

                  {isDispatch && (
                    <span className="h-1.5 w-1.5 rounded-full bg-[#E12D1E]" />
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-5">
            <div className="hidden items-center gap-2.5 font-mono text-[9.5px] uppercase tracking-[0.24em] text-[var(--oculus-muted)] xl:flex">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--oculus-ink)]" />
              <span>STT · LLAMA 3.3</span>
            </div>

            <div className="opacity-75 transition-opacity hover:opacity-100">
              <ThemeSwitcher />
            </div>

            <MagneticLink
              href="/sessions/new"
              className="group hidden items-center gap-2.5 bg-[var(--oculus-ink)] px-5 py-2.5 font-mono text-[10px] font-medium uppercase tracking-[0.26em] text-[var(--oculus-bg)] transition-opacity hover:opacity-88 sm:inline-flex"
            >
              <span>+ LAUNCH</span>
              <span className="transition-transform duration-300 group-hover:translate-x-0.5">
                →
              </span>
            </MagneticLink>

            <button
              type="button"
              onClick={() => setMenuOpen((o) => !o)}
              aria-label="Toggle navigation menu"
              aria-expanded={menuOpen}
              className="flex h-9 w-9 flex-col items-end justify-center gap-1.5 md:hidden focus:outline-none"
            >
              <span
                className={`h-[1.5px] bg-[var(--oculus-ink)] transition-all duration-300 ${
                  menuOpen ? "w-6 translate-y-[3.5px] rotate-45" : "w-6"
                }`}
              />
              <span
                className={`h-[1.5px] bg-[var(--oculus-ink)] transition-all duration-300 ${
                  menuOpen ? "w-6 -translate-y-[4px] -rotate-45" : "w-4"
                }`}
              />
            </button>
          </div>
        </div>
      </header>

      <div
        aria-hidden={!menuOpen}
        className={`fixed inset-0 z-40 flex flex-col justify-between bg-[var(--oculus-ink)] px-8 pb-12 pt-28 text-[var(--oculus-bg)] transition-all duration-700 ease-[cubic-bezier(0.19,1,0.22,1)] md:hidden ${
          menuOpen
            ? "pointer-events-auto translate-y-0 opacity-100"
            : "pointer-events-none -translate-y-full opacity-0"
        }`}
      >
        <nav className="flex flex-col gap-6">
          {NAV_ITEMS.map((item, idx) => (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMenuOpen(false)}
              style={{
                transitionDelay: menuOpen ? `${120 + idx * 70}ms` : "0ms",
              }}
              className={`group flex items-baseline justify-between border-b border-[var(--oculus-bg)]/15 pb-5 transition-all duration-500 ${
                menuOpen
                  ? "translate-y-0 opacity-100"
                  : "translate-y-6 opacity-0"
              }`}
            >
              <div className="flex items-baseline gap-4">
                <span className="font-mono text-xs tracking-[0.3em] opacity-45">
                  {item.index}
                </span>
                <span className="font-display text-3xl font-semibold tracking-[-0.03em]">
                  {item.label}
                </span>
              </div>
              <span className="font-mono text-sm opacity-60 transition-transform group-hover:translate-x-1">
                →
              </span>
            </Link>
          ))}
        </nav>

        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.26em] opacity-55">
          <span>OCULUS // SENTINEL</span>
          <span>Q0 CALIBRATED</span>
        </div>
      </div>
    </>
  );
}
