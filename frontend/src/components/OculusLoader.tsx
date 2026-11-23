"use client";

import React, { useEffect, useRef, useState } from "react";
import gsap from "gsap";

const UPPER_PTS: [number, number][] = [
  [420, 51],
  [470, 7],
  [530, 7],
  [580, 54],
  [552, 71],
  [518, 103],
  [482, 103],
  [447, 69],
];

const LOWER_PTS: [number, number][] = [
  [580, 89],
  [530, 133],
  [470, 133],
  [420, 86],
  [448, 69],
  [482, 37],
  [518, 37],
  [553, 71],
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function buildRibbonPath(
  pts: [number, number][],
  collapse: number,
  extend: number
) {
  const m = pts.map(([bx, by]) => {
    const dx = bx - 500;
    const xFlat = lerp(bx, 500 + dx * 1.55, collapse);
    const x = lerp(xFlat, 500 + (dx / 80) * 2100, extend);
    const y = lerp(by, 70, collapse);
    return [x.toFixed(1), y.toFixed(1)];
  });

  return `M ${m[0][0]} ${m[0][1]} C ${m[1][0]} ${m[1][1]}, ${m[2][0]} ${m[2][1]}, ${m[3][0]} ${m[3][1]} L ${m[4][0]} ${m[4][1]} C ${m[5][0]} ${m[5][1]}, ${m[6][0]} ${m[6][1]}, ${m[7][0]} ${m[7][1]}`;
}

export default function OculusLoader({
  oncePerSession = false,
  onComplete,
}: {
  oncePerSession?: boolean;
  onComplete?: () => void;
}) {
  const [mounted, setMounted] = useState(true);
  const [phase, setPhase] = useState("OPTICAL INIT");
  const [pct, setPct] = useState(0);

  const rootRef = useRef<HTMLDivElement>(null);
  const topCurtainRef = useRef<HTMLDivElement>(null);
  const bottomCurtainRef = useRef<HTMLDivElement>(null);
  const topSeamRef = useRef<HTMLDivElement>(null);
  const bottomSeamRef = useRef<HTMLDivElement>(null);
  const svgWrapRef = useRef<HTMLDivElement>(null);
  const upperPathRef = useRef<SVGPathElement>(null);
  const lowerPathRef = useRef<SVGPathElement>(null);
  const pupilRef = useRef<SVGEllipseElement>(null);
  const pulseRef = useRef<SVGCircleElement>(null);
  const ringRef = useRef<SVGGElement>(null);
  const hudRef = useRef<HTMLDivElement>(null);
  const tlRef = useRef<gsap.core.Timeline | null>(null);

  const finish = () => {
    document.body.style.overflow = "";
    if (oncePerSession) sessionStorage.setItem("oculus_loader_done", "1");
    setMounted(false);
    onComplete?.();
  };

  useEffect(() => {
    if (oncePerSession && sessionStorage.getItem("oculus_loader_done") === "1") {
      setMounted(false);
      onComplete?.();
      return;
    }

    document.body.style.overflow = "hidden";

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const upper = upperPathRef.current;
    const lower = lowerPathRef.current;

    if (!upper || !lower) return;

    const uLen = upper.getTotalLength();
    const lLen = lower.getTotalLength();

    gsap.set([upper, lower], {
      strokeDasharray: (i) => (i === 0 ? uLen : lLen),
      strokeDashoffset: (i) => (i === 0 ? uLen : lLen),
    });

    const state = {
      collapse: 0,
      extend: 0,
      stroke: 12.5,
      progress: 0,
    };

    const syncGeometry = () => {
      upper.setAttribute(
        "d",
        buildRibbonPath(UPPER_PTS, state.collapse, state.extend)
      );
      lower.setAttribute(
        "d",
        buildRibbonPath(LOWER_PTS, state.collapse, state.extend)
      );
      upper.setAttribute("stroke-width", state.stroke.toFixed(2));
      lower.setAttribute("stroke-width", state.stroke.toFixed(2));
    };

    const tl = gsap.timeline({
      onComplete: finish,
    });
    tlRef.current = tl;

    if (reduced) {
      tl.to(topCurtainRef.current, { yPercent: -100, duration: 0.45, ease: "power3.inOut" }, 0.1)
        .to(bottomCurtainRef.current, { yPercent: 100, duration: 0.45, ease: "power3.inOut" }, 0.1)
        .to([svgWrapRef.current, hudRef.current], { opacity: 0, duration: 0.2 }, 0);
      return () => {
        tl.kill();
        document.body.style.overflow = "";
      };
    }

    tl.to(
      state,
      {
        progress: 100,
        duration: 1.55,
        ease: "power2.inOut",
        onUpdate: () => setPct(Math.round(state.progress)),
      },
      0
    )
      .fromTo(
        ringRef.current,
        { opacity: 0, scale: 0.82, rotate: -38, transformOrigin: "500px 70px" },
        {
          opacity: 0.32,
          scale: 1,
          rotate: 0,
          duration: 0.9,
          ease: "expo.out",
        },
        0.05
      )
      .to(
        [upper, lower],
        {
          strokeDashoffset: 0,
          duration: 0.95,
          stagger: 0.08,
          ease: "expo.inOut",
        },
        0.1
      )
      .fromTo(
        pupilRef.current,
        { attr: { rx: 0, ry: 0 }, opacity: 0 },
        {
          attr: { rx: 11.8, ry: 11.8 },
          opacity: 1,
          duration: 0.55,
          ease: "back.out(1.7)",
        },
        0.58
      )
      .fromTo(
        pulseRef.current,
        { attr: { r: 12 }, opacity: 0.85 },
        { attr: { r: 64 }, opacity: 0, duration: 0.7, ease: "expo.out" },
        0.72
      )
      .call(() => setPhase("CONVERGING 1-AXIS"), undefined, 1.18)
      .set([upper, lower], { strokeDasharray: "none" }, 1.18)
      .to(
        ringRef.current,
        {
          scaleY: 0,
          opacity: 0,
          duration: 0.42,
          ease: "expo.in",
          transformOrigin: "500px 70px",
        },
        1.18
      )
      .to(
        pupilRef.current,
        {
          attr: { rx: 68, ry: 0.7 },
          duration: 0.48,
          ease: "expo.inOut",
        },
        1.22
      )
      .to(
        state,
        {
          collapse: 1,
          stroke: 3.2,
          duration: 0.52,
          ease: "expo.inOut",
          onUpdate: syncGeometry,
        },
        1.22
      )
      .call(() => setPhase("AXIS LOCK // CLEAVING"), undefined, 1.64)
      .to(
        pupilRef.current,
        {
          attr: { rx: 1400, ry: 0.4 },
          opacity: 0,
          duration: 0.44,
          ease: "expo.in",
        },
        1.64
      )
      .to(
        state,
        {
          extend: 1,
          stroke: 1.4,
          duration: 0.5,
          ease: "expo.inOut",
          onUpdate: syncGeometry,
        },
        1.62
      )
      .to(
        hudRef.current,
        {
          opacity: 0,
          duration: 0.25,
          ease: "power2.out",
        },
        1.92
      )
      .set([topSeamRef.current, bottomSeamRef.current], { opacity: 1 }, 2.12)
      .set(svgWrapRef.current, { opacity: 0 }, 2.12)
      .to(
        topCurtainRef.current,
        {
          yPercent: -100,
          duration: 0.95,
          ease: "expo.inOut",
        },
        2.16
      )
      .to(
        bottomCurtainRef.current,
        {
          yPercent: 100,
          duration: 0.95,
          ease: "expo.inOut",
        },
        2.16
      );

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        tl.kill();
        finish();
      }
    };
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("keydown", onKey);
      tl.kill();
      document.body.style.overflow = "";
    };
  }, [oncePerSession]);

  if (!mounted) return null;

  const ticks = Array.from({ length: 72 }, (_, i) => {
    const rad = (i * 5 * Math.PI) / 180;
    const major = i % 6 === 0;
    const r1 = major ? 78 : 82;
    const r2 = 86;
    return {
      x1: 500 + Math.cos(rad) * r1,
      y1: 70 + Math.sin(rad) * r1,
      x2: 500 + Math.cos(rad) * r2,
      y2: 70 + Math.sin(rad) * r2,
      major,
    };
  });

  return (
    <div
      ref={rootRef}
      aria-live="polite"
      className="fixed inset-0 z-[100] overflow-hidden select-none"
    >
      <div
        ref={topCurtainRef}
        className="absolute inset-x-0 top-0 h-1/2 bg-[var(--oculus-bg,#FBFBF9)] will-change-transform"
      >
        <div className="mx-auto flex h-full max-w-[1380px] flex-col justify-between border-x border-[var(--oculus-hairline,rgba(10,10,10,0.12))] px-5 py-6 sm:px-8">
          <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--oculus-muted,#6E6E6A)]">
            <span className="flex items-center gap-2.5 text-[var(--oculus-ink,#0A0A0A)]">
              <span className="inline-block h-1.5 w-1.5 bg-current" />
              OCULUS // SENTINEL
            </span>
            <span className="tabular-nums">Q0 BASELINE</span>
          </div>
        </div>
        <div
          ref={topSeamRef}
          className="absolute inset-x-0 bottom-0 h-[1px] bg-[var(--oculus-ink,#0A0A0A)] opacity-0"
        />
      </div>

      <div
        ref={bottomCurtainRef}
        className="absolute inset-x-0 bottom-0 h-1/2 bg-[var(--oculus-bg,#FBFBF9)] will-change-transform"
      >
        <div
          ref={bottomSeamRef}
          className="absolute inset-x-0 top-0 h-[1px] bg-[var(--oculus-ink,#0A0A0A)] opacity-0"
        />
        <div className="mx-auto flex h-full max-w-[1380px] flex-col justify-end border-x border-[var(--oculus-hairline,rgba(10,10,10,0.12))] px-5 py-6 sm:px-8">
          <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--oculus-muted,#6E6E6A)]">
            <span>MULTIMODAL INTEGRITY ENGINE</span>
            <button
              type="button"
              onClick={() => {
                tlRef.current?.kill();
                finish();
              }}
              className="pointer-events-auto text-[var(--oculus-ink,#0A0A0A)] transition-opacity hover:opacity-50"
            >
              [ SKIP ]
            </button>
          </div>
        </div>
      </div>

      <div
        ref={svgWrapRef}
        className="pointer-events-none absolute inset-0 flex items-center justify-center"
      >
        <svg
          viewBox="0 0 1000 140"
          className="h-[140px] w-[1000px] max-w-none scale-[0.82] overflow-visible text-[var(--oculus-ink,#0A0A0A)] sm:scale-100 md:scale-110"
        >
          <g ref={ringRef} opacity="0">
            <circle
              cx="500"
              cy="70"
              r="86"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.45"
            />
            {ticks.map((t, i) => (
              <line
                key={i}
                x1={t.x1}
                y1={t.y1}
                x2={t.x2}
                y2={t.y2}
                stroke="currentColor"
                strokeWidth={t.major ? "0.9" : "0.4"}
              />
            ))}
          </g>

          <circle
            ref={pulseRef}
            cx="500"
            cy="70"
            r="12"
            fill="none"
            stroke="currentColor"
            strokeWidth="0.9"
            opacity="0"
          />

          <path
            ref={upperPathRef}
            d={buildRibbonPath(UPPER_PTS, 0, 0)}
            fill="none"
            stroke="currentColor"
            strokeWidth="12.5"
            strokeLinecap="butt"
            strokeLinejoin="miter"
            strokeMiterlimit="3"
          />

          <path
            ref={lowerPathRef}
            d={buildRibbonPath(LOWER_PTS, 0, 0)}
            fill="none"
            stroke="currentColor"
            strokeWidth="12.5"
            strokeLinecap="butt"
            strokeLinejoin="miter"
            strokeMiterlimit="3"
          />

          <ellipse
            ref={pupilRef}
            cx="500"
            cy="70"
            rx="0"
            ry="0"
            fill="currentColor"
          />
        </svg>
      </div>

      <div
        ref={hudRef}
        className="pointer-events-none absolute inset-x-0 top-1/2 mt-24 flex flex-col items-center gap-2 font-mono text-[10px] uppercase tracking-[0.32em] text-[var(--oculus-muted,#6E6E6A)]"
      >
        <div className="flex items-center gap-3 text-[var(--oculus-ink,#0A0A0A)] tabular-nums">
          <span className="h-1.5 w-1.5 rounded-full bg-[#E12D1E]" />
          <span>{phase}</span>
          <span className="opacity-40">//</span>
          <span>{String(pct).padStart(3, "0")}%</span>
        </div>
      </div>
    </div>
  );
}
