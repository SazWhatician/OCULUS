"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import ThemeSwitcher from "@/components/ThemeSwitcher";

const NAV_ITEMS = [
  { href: "/", code: "01", label: "DASHBOARD", bearing: 0, meta: "SIGNAL MESH & LEDGER" },
  { href: "/sessions/new", code: "02", label: "DISPATCH", bearing: 120, meta: "INITIALIZE SENTINEL" },
  { href: "/reports", code: "03", label: "AUDIT ARCHIVE", bearing: 240, meta: "CALIBRATED VERDICTS" },
];

function normalizeAngle(deg: number) {
  return ((deg % 360) + 360) % 360;
}

function angularDist(a: number, b: number) {
  const d = Math.abs(normalizeAngle(a) - normalizeAngle(b));
  return d > 180 ? 360 - d : d;
}

function VernierRuler({ progress }: { progress: number }) {
  const ticks = Array.from({ length: 61 }, (_, i) => i);
  const shift = -progress * 240;

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-x-0 top-0 z-[60] h-[11px] overflow-hidden"
    >
      <div
        className="mx-auto flex h-full w-[760px] items-start justify-between transition-transform duration-150 ease-out will-change-transform"
        style={{
          transform: `translate3d(${shift}px, 0, 0)`,
          maskImage:
            "linear-gradient(to right, transparent 0%, black 18%, black 82%, transparent 100%)",
          WebkitMaskImage:
            "linear-gradient(to right, transparent 0%, black 18%, black 82%, transparent 100%)",
        }}
      >
        {ticks.map((i) => {
          const major = i % 10 === 0;
          const mid = i % 5 === 0 && !major;
          return (
            <span
              key={i}
              className="block w-px bg-[var(--oculus-ink)]"
              style={{
                height: major ? 8 : mid ? 5 : 3,
                opacity: major ? 0.55 : mid ? 0.28 : 0.14,
              }}
            />
          );
        })}
      </div>
      <span className="absolute left-1/2 top-0 h-[10px] w-[1.5px] -translate-x-1/2 bg-[#E12D1E]" />
    </div>
  );
}

function KineticOculusCore({
  dialOpen,
  onToggle,
}: {
  dialOpen: boolean;
  onToggle: () => void;
}) {
  const pupilRef = useRef<SVGCircleElement>(null);
  const ringRef = useRef<SVGGElement>(null);
  const sRef = useRef({ x: 0, y: 0, vx: 0, vy: 0, tx: 0, ty: 0, rot: 0 });

  useEffect(() => {
    let raf = 0;

    const onMove = (e: MouseEvent) => {
      const nx = (e.clientX / window.innerWidth - 0.14) * 2;
      const ny = (e.clientY / window.innerHeight - 0.08) * 2;
      const a = Math.atan2(ny, nx);
      const m = Math.min(Math.hypot(nx, ny), 1);
      sRef.current.tx = Math.cos(a) * 32 * m;
      sRef.current.ty = Math.sin(a) * 14 * m;
    };

    window.addEventListener("mousemove", onMove, { passive: true });

    const tick = () => {
      const s = sRef.current;
      s.vx = (s.vx + (s.tx - s.x) * 0.11) * 0.74;
      s.vy = (s.vy + (s.ty - s.y) * 0.11) * 0.74;
      s.x += s.vx;
      s.y += s.vy;
      s.rot += 0.15 + Math.hypot(s.vx, s.vy) * 1.4;

      if (pupilRef.current) {
        pupilRef.current.setAttribute("cx", (240 + s.x).toFixed(2));
        pupilRef.current.setAttribute("cy", (180 + s.y).toFixed(2));
      }
      if (ringRef.current) {
        ringRef.current.setAttribute("transform", `rotate(${s.rot.toFixed(1)} 240 180)`);
      }
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("mousemove", onMove);
    };
  }, []);

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label="Toggle Polar Optical Command Dial"
      aria-expanded={dialOpen}
      className="group relative flex items-center gap-3.5 focus:outline-none"
    >
      <svg
        viewBox="-40 -40 560 440"
        className="h-9 w-12 overflow-visible text-[var(--oculus-ink)]"
      >
        <g ref={ringRef} opacity="0.28">
          <circle
            cx="240"
            cy="180"
            r="165"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="6 16"
          />
        </g>
        <path
          fillRule="evenodd"
          d="M2,102L1,152L150,66L171,56L192,49L216,44L256,43L293,50L315,58L438,128L439,130L404,151L395,154L305,102L290,95L270,89L249,86L218,88L205,91L182,100L1,204L1,257L129,331L156,344L176,351L215,359L258,360L282,357L311,350L351,333L479,259L479,209L336,292L314,303L295,310L267,316L225,317L192,311L167,302L44,232L44,229L86,205L177,258L200,268L232,274L265,272L287,266L306,257L478,158L479,102L353,29L333,19L309,10L280,3L255,0L215,1L172,10L130,28Z M130,180L209,135L234,129L257,130L281,138L349,177L352,180L274,225L249,231L225,230L205,224L133,183Z"
          fill="currentColor"
          className="transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:scale-[1.02]"
        />
        <circle
          ref={pupilRef}
          cx="240"
          cy="180"
          r={dialOpen ? 44 : 36.4}
          fill={dialOpen ? "#E12D1E" : "currentColor"}
          className="transition-colors duration-300"
        />
      </svg>

      <div className="flex flex-col text-left leading-none">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-[17px] font-semibold tracking-[0.08em] text-[var(--oculus-ink)]">
            OCULUS
          </span>
          <span className="font-mono text-[8.5px] uppercase tracking-[0.3em] text-[var(--oculus-muted)]">
            {dialOpen ? "DIAL ACTIVE" : "⌘K DIAL"}
          </span>
        </div>
        <span className="mt-1 hidden font-mono text-[8.5px] uppercase tracking-[0.26em] text-[var(--oculus-muted)] sm:block">
          OPTICAL INTEGRITY SENTINEL
        </span>
      </div>
    </button>
  );
}

function ApertureRail({ pathname }: { pathname: string }) {
  const railRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLAnchorElement | null)[]>([]);
  const svgRef = useRef<SVGSVGElement>(null);
  const upperRef = useRef<SVGPathElement>(null);
  const lowerRef = useRef<SVGPathElement>(null);
  const gradRef = useRef<SVGLinearGradientElement>(null);
  const apexRef = useRef<SVGCircleElement>(null);
  const nadirRef = useRef<SVGCircleElement>(null);

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [railWidth, setRailWidth] = useState(460);

  const activeIdx = NAV_ITEMS.findIndex((item) =>
    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
  );
  const targetIdx = hoveredIdx !== null ? hoveredIdx : activeIdx;

  const physRef = useRef({
    x: 230,
    vx: 0,
    tx: 230,
    w: 110,
    vw: 0,
    tw: 110,
    open: 0,
    vOpen: 0,
    tOpen: 0,
  });

  const measureTarget = useCallback(() => {
    const rail = railRef.current;
    if (!rail) return;
    const rRect = rail.getBoundingClientRect();
    setRailWidth(rRect.width);

    const el = targetIdx >= 0 ? itemRefs.current[targetIdx] : null;
    if (!el) {
      physRef.current.tOpen = 0;
      return;
    }
    const eRect = el.getBoundingClientRect();
    physRef.current.tx = eRect.left - rRect.left + eRect.width / 2;
    physRef.current.tw = eRect.width;
    physRef.current.tOpen = hoveredIdx !== null ? 1 : 0.72;
  }, [targetIdx, hoveredIdx]);

  useEffect(() => {
    measureTarget();
    window.addEventListener("resize", measureTarget);
    return () => window.removeEventListener("resize", measureTarget);
  }, [measureTarget, pathname]);

  useEffect(() => {
    let raf = 0;

    const tick = () => {
      const p = physRef.current;
      const k = 0.14;
      const d = 0.74;

      p.vx = (p.vx + (p.tx - p.x) * k) * d;
      p.x += p.vx;

      p.vw = (p.vw + (p.tw - p.w) * k) * d;
      p.w += p.vw;

      p.vOpen = (p.vOpen + (p.tOpen - p.open) * 0.12) * 0.76;
      p.open += p.vOpen;

      const W = Math.max(railWidth, 320);
      const base = 44;
      const cx = p.x;
      const hw = p.w * 0.56 + 22;
      const span = hw * 1.48;
      const leftX = cx - span;
      const rightX = cx + span;
      const shear = Math.max(Math.min(p.vx * 1.15, 16), -16);

      const upperCtrlY = base - 54 * p.open;
      const lowerCtrlY = base + 14 * p.open;

      const upperD = `M 0 ${base} L ${leftX.toFixed(1)} ${base} C ${(cx - hw * 0.62 + shear).toFixed(1)} ${upperCtrlY.toFixed(1)}, ${(cx + hw * 0.62 + shear).toFixed(1)} ${upperCtrlY.toFixed(1)}, ${rightX.toFixed(1)} ${base} L ${W} ${base}`;
      const lowerD = `M 0 ${base} L ${leftX.toFixed(1)} ${base} C ${(cx - hw * 0.52 - shear * 0.5).toFixed(1)} ${lowerCtrlY.toFixed(1)}, ${(cx + hw * 0.52 - shear * 0.5).toFixed(1)} ${lowerCtrlY.toFixed(1)}, ${rightX.toFixed(1)} ${base} L ${W} ${base}`;

      if (upperRef.current) upperRef.current.setAttribute("d", upperD);
      if (lowerRef.current) lowerRef.current.setAttribute("d", lowerD);

      if (gradRef.current) {
        gradRef.current.setAttribute("x1", (cx - 210).toFixed(1));
        gradRef.current.setAttribute("x2", (cx + 210).toFixed(1));
      }

      const apexY = 0.25 * base + 0.75 * upperCtrlY;
      const nadirY = 0.25 * base + 0.75 * lowerCtrlY;

      if (apexRef.current) {
        apexRef.current.setAttribute("cx", (cx + shear * 0.35).toFixed(1));
        apexRef.current.setAttribute("cy", apexY.toFixed(1));
        apexRef.current.setAttribute("opacity", (p.open * 0.95).toFixed(2));
      }
      if (nadirRef.current) {
        nadirRef.current.setAttribute("cx", (cx - shear * 0.2).toFixed(1));
        nadirRef.current.setAttribute("cy", nadirY.toFixed(1));
        nadirRef.current.setAttribute("opacity", (p.open * 0.45).toFixed(2));
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [railWidth]);

  return (
    <div
      ref={railRef}
      onMouseLeave={() => setHoveredIdx(null)}
      className="relative hidden h-[58px] items-center gap-12 px-8 md:flex"
    >
      <svg
        ref={svgRef}
        width={railWidth}
        height={58}
        viewBox={`0 0 ${railWidth} 58`}
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 overflow-visible text-[var(--oculus-ink)]"
      >
        <defs>
          <linearGradient
            ref={gradRef}
            id="oculus-horizon-fade"
            gradientUnits="userSpaceOnUse"
            x1="0"
            y1="0"
            x2="420"
            y2="0"
          >
            <stop offset="0%" stopColor="currentColor" stopOpacity="0" />
            <stop offset="22%" stopColor="currentColor" stopOpacity="0.22" />
            <stop offset="50%" stopColor="currentColor" stopOpacity="0.92" />
            <stop offset="78%" stopColor="currentColor" stopOpacity="0.22" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>

        <path
          ref={upperRef}
          fill="none"
          stroke="url(#oculus-horizon-fade)"
          strokeWidth="1.25"
        />
        <path
          ref={lowerRef}
          fill="none"
          stroke="url(#oculus-horizon-fade)"
          strokeWidth="0.95"
        />
        <circle ref={apexRef} cx="200" cy="6" r="2" fill="#E12D1E" opacity="0" />
        <circle ref={nadirRef} cx="200" cy="54" r="1.4" fill="currentColor" opacity="0" />
      </svg>

      {NAV_ITEMS.map((item, idx) => {
        const isActive = idx === activeIdx;
        return (
          <Link
            key={item.href}
            ref={(el) => {
              itemRefs.current[idx] = el;
            }}
            href={item.href}
            onMouseEnter={() => setHoveredIdx(idx)}
            className={`group relative z-10 flex items-center gap-2 py-2 font-mono text-[10.5px] uppercase tracking-[0.28em] transition-opacity duration-300 focus:outline-none ${isActive
              ? "text-[var(--oculus-ink)] opacity-100"
              : "text-[var(--oculus-ink)] opacity-60 hover:opacity-100"
              }`}
          >
            <span className="text-[8.5px] opacity-45">{item.code}</span>
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
          </Link>
        );
      })}
    </div>
  );
}

function PolarAzimuthDial({
  open,
  onClose,
  pathname,
}: {
  open: boolean;
  onClose: () => void;
  pathname: string;
}) {
  const router = useRouter();
  const [angle, setAngle] = useState(0);
  const [lockedIdx, setLockedIdx] = useState(0);
  const needleRef = useRef<SVGGElement>(null);
  const smoothRef = useRef({ cur: 0, target: 0 });

  useEffect(() => {
    if (!open) return;
    let raf = 0;

    const onMove = (e: MouseEvent) => {
      const cx = window.innerWidth / 2;
      const cy = window.innerHeight / 2;
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;
      if (Math.hypot(dx, dy) < 28) return;

      const rawDeg = normalizeAngle((Math.atan2(dy, dx) * 180) / Math.PI + 90);
      smoothRef.current.target = rawDeg;
      setAngle(rawDeg);

      let best = 0;
      let minD = 360;
      NAV_ITEMS.forEach((item, i) => {
        const d = angularDist(rawDeg, item.bearing);
        if (d < minD) {
          minD = d;
          best = i;
        }
      });
      setLockedIdx(best);
    };

    const tick = () => {
      const s = smoothRef.current;
      let diff = s.target - s.cur;
      if (diff > 180) diff -= 360;
      if (diff < -180) diff += 360;
      s.cur = normalizeAngle(s.cur + diff * 0.18);

      if (needleRef.current) {
        needleRef.current.setAttribute(
          "transform",
          `rotate(${s.cur.toFixed(2)} 260 260)`
        );
      }
      raf = requestAnimationFrame(tick);
    };

    window.addEventListener("mousemove", onMove, { passive: true });
    raf = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(raf);
    };
  }, [open]);

  if (!open) return null;

  const navigateTo = (href: string) => {
    onClose();
    router.push(href);
  };

  const dialTicks = Array.from({ length: 72 }, (_, i) => {
    const rad = ((i * 5 - 90) * Math.PI) / 180;
    const major = i % 6 === 0;
    const r1 = major ? 186 : 194;
    const r2 = 202;
    return {
      x1: 260 + Math.cos(rad) * r1,
      y1: 260 + Math.sin(rad) * r1,
      x2: 260 + Math.cos(rad) * r2,
      y2: 260 + Math.sin(rad) * r2,
      major,
    };
  });

  const locked = NAV_ITEMS[lockedIdx];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="OCULUS Polar Navigation Dial"
      className="fixed inset-0 z-[90] flex flex-col items-center justify-center bg-[var(--oculus-ink)] text-[var(--oculus-bg)] transition-opacity duration-300"
    >
      <div className="pointer-events-none absolute inset-x-0 top-7 mx-auto flex max-w-[1380px] items-center justify-between px-6 font-mono text-[10px] uppercase tracking-[0.3em] opacity-60 sm:px-10">
        <span>POLAR AZIMUTH // COMMAND INSTRUMENT</span>
        <span className="tabular-nums">
          BEARING {angle.toFixed(1).padStart(5, "0")}°
        </span>
      </div>

      <div className="relative flex items-center justify-center">
        <svg
          viewBox="0 0 520 520"
          className="h-[340px] w-[340px] overflow-visible sm:h-[480px] sm:w-[480px]"
        >
          <circle
            cx="260"
            cy="260"
            r="202"
            fill="none"
            stroke="currentColor"
            strokeOpacity="0.18"
            strokeWidth="0.75"
          />
          <circle
            cx="260"
            cy="260"
            r="142"
            fill="none"
            stroke="currentColor"
            strokeOpacity="0.12"
            strokeWidth="0.75"
            strokeDasharray="3 7"
          />

          {dialTicks.map((t, i) => (
            <line
              key={i}
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke="currentColor"
              strokeOpacity={t.major ? 0.55 : 0.2}
              strokeWidth={t.major ? 1.1 : 0.5}
            />
          ))}

          <g ref={needleRef}>
            <line
              x1="260"
              y1="260"
              x2="260"
              y2="58"
              stroke="#E12D1E"
              strokeWidth="1.2"
            />
            <circle cx="260" cy="58" r="3.5" fill="#E12D1E" />
          </g>

          <g transform="translate(195 215) scale(0.65)">
            <path
              d="M 20 51 C 70 7, 130 7, 180 54 L 152 71 C 118 103, 82 103, 47 69"
              fill="none"
              stroke="currentColor"
              strokeWidth="11"
            />
            <path
              d="M 180 89 C 130 133, 70 133, 20 86 L 48 69 C 82 37, 118 37, 153 71"
              fill="none"
              stroke="currentColor"
              strokeWidth="11"
            />
            <circle cx="100" cy="70" r="11" fill="currentColor" />
          </g>
        </svg>

        {NAV_ITEMS.map((item, idx) => {
          const rad = ((item.bearing - 90) * Math.PI) / 180;
          const isLocked = idx === lockedIdx;
          const isCurrent =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

          return (
            <button
              key={item.href}
              type="button"
              onClick={() => navigateTo(item.href)}
              onMouseEnter={() => setLockedIdx(idx)}
              style={{
                left: `${50 + Math.cos(rad) * 44}%`,
                top: `${50 + Math.sin(rad) * 44}%`,
              }}
              className={`group absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap px-4 py-2 text-center transition-all duration-300 focus:outline-none ${isLocked ? "scale-105 opacity-100" : "opacity-45 hover:opacity-100"
                }`}
            >
              <span className="block font-mono text-[9px] tracking-[0.3em] opacity-60">
                {String(item.bearing).padStart(3, "0")}° // {item.code}
                {isCurrent ? " · ACTIVE" : ""}
              </span>
              <span className="mt-1 block font-display text-xl font-semibold tracking-[-0.02em] sm:text-2xl">
                {item.label}
              </span>
              <span className="mt-0.5 block font-mono text-[8.5px] tracking-[0.22em] opacity-55">
                {item.meta}
              </span>
            </button>
          );
        })}
      </div>

      <div className="absolute inset-x-0 bottom-8 mx-auto flex max-w-[1380px] flex-wrap items-center justify-between gap-4 px-6 font-mono text-[10px] uppercase tracking-[0.28em] sm:px-10">
        <button
          type="button"
          onClick={() => navigateTo(locked.href)}
          className="border border-current/35 px-4 py-2 transition-colors hover:bg-[var(--oculus-bg)] hover:text-[var(--oculus-ink)]"
        >
          ENTER → {locked.label}
        </button>

        <ThemeSwitcher />

        <button
          type="button"
          onClick={onClose}
          className="opacity-65 transition-opacity hover:opacity-100"
        >
          [ ESC TO CLOSE ]
        </button>
      </div>
    </div>
  );
}

export default function OculusNavbar() {
  const pathname = usePathname();
  const [scrollY, setScrollY] = useState(0);
  const [squint, setSquint] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dialOpen, setDialOpen] = useState(false);

  const lastYRef = useRef(0);

  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(Math.max(y / max, 0), 1) : 0);
      setScrollY(y);
      setSquint(y > 90 && y > lastYRef.current);
      lastYRef.current = y;
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setDialOpen((o) => !o);
      } else if (e.key === "Escape") {
        setDialOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    setDialOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = dialOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [dialOpen]);

  return (
    <>
      <VernierRuler progress={progress} />

      <header
        onMouseEnter={() => setSquint(false)}
        className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] ${squint ? "pt-3 pb-1" : scrollY > 18 ? "pt-4 pb-2" : "pt-6 pb-3"
          }`}
      >
        <div
          aria-hidden="true"
          className={`pointer-events-none absolute inset-x-0 -bottom-10 top-0 transition-opacity duration-500 ${scrollY > 12 ? "opacity-100" : "opacity-0"
            }`}
          style={{
            background:
              "linear-gradient(to bottom, var(--oculus-bg) 0%, color-mix(in srgb, var(--oculus-bg) 82%, transparent) 58%, transparent 100%)",
            backdropFilter: "blur(14px)",
            WebkitBackdropFilter: "blur(14px)",
            maskImage:
              "linear-gradient(to bottom, black 62%, transparent 100%)",
            WebkitMaskImage:
              "linear-gradient(to bottom, black 62%, transparent 100%)",
          }}
        />

        <div className="relative z-10 mx-auto flex max-w-[1380px] items-center justify-between px-5 sm:px-8">
          <KineticOculusCore
            dialOpen={dialOpen}
            onToggle={() => setDialOpen((o) => !o)}
          />

          <ApertureRail pathname={pathname} />

          <div className="flex items-center gap-4">
            <span
              className={`hidden font-mono text-[9px] uppercase tracking-[0.26em] text-[var(--oculus-muted)] tabular-nums transition-opacity duration-500 xl:inline ${squint ? "opacity-25" : "opacity-80"
                }`}
            >
              DEPTH {String(Math.round(progress * 100)).padStart(3, "0")}%
            </span>

            <div className="opacity-75 transition-opacity hover:opacity-100">
              <ThemeSwitcher />
            </div>

            <Link
              href="/sessions/new"
              className="group relative hidden items-center gap-2.5 overflow-hidden bg-[var(--oculus-ink)] px-4 py-2 font-mono text-[9.5px] font-medium uppercase tracking-[0.26em] text-[var(--oculus-bg)] transition-opacity hover:opacity-85 sm:inline-flex"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-[#E12D1E]" />
              <span>LAUNCH</span>
              <span className="transition-transform duration-300 group-hover:translate-x-0.5">
                →
              </span>
            </Link>

            <button
              type="button"
              onClick={() => setDialOpen(true)}
              className="font-mono text-[10px] uppercase tracking-[0.26em] text-[var(--oculus-ink)] md:hidden"
            >
              [ DIAL ]
            </button>
          </div>
        </div>
      </header>

      <PolarAzimuthDial
        open={dialOpen}
        onClose={() => setDialOpen(false)}
        pathname={pathname}
      />
    </>
  );
}