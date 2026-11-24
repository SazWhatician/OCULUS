"use client";

import React, { useEffect, useRef, useState } from "react";

export interface GazeTelemetry {
  yaw: number;
  pitch: number;
  dilation: number;
  velocity: number;
}

export function OpticalFieldCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    let width = 0;
    let height = 0;
    const pointer = { x: -1000, y: -1000, tx: -1000, ty: -1000, active: false };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const onMove = (e: MouseEvent) => {
      pointer.tx = e.clientX;
      pointer.ty = e.clientY;
      pointer.active = true;
    };

    const onLeave = () => {
      pointer.active = false;
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseleave", onLeave);

    const spacing = 64;
    const radius = 240;

    const displace = (x: number, y: number) => {
      const dx = x - pointer.x;
      const dy = y - pointer.y;
      const distSq = dx * dx + dy * dy;
      if (distSq > radius * radius || distSq < 0.001) return { x, y };
      const dist = Math.sqrt(distSq);
      const factor = (1 - dist / radius) ** 2 * 18;
      return {
        x: x + (dx / dist) * factor,
        y: y + (dy / dist) * factor,
      };
    };

    const isDark = () => {
      if (typeof document === "undefined") return false;
      const theme = document.documentElement.getAttribute("data-theme");
      return (
        theme === "dark" ||
        theme === "obsidian" ||
        theme === "ocean" ||
        document.documentElement.classList.contains("dark")
      );
    };

    const render = () => {
      const dark = isDark();

      if (pointer.active) {
        pointer.x += (pointer.tx - pointer.x) * 0.1;
        pointer.y += (pointer.ty - pointer.y) * 0.1;
      } else {
        pointer.x += (-1000 - pointer.x) * 0.04;
        pointer.y += (-1000 - pointer.y) * 0.04;
      }

      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = dark
        ? "rgba(251, 251, 249, 0.055)"
        : "rgba(10, 10, 10, 0.045)";
      ctx.lineWidth = 1;

      const cols = Math.ceil(width / spacing) + 1;
      const rows = Math.ceil(height / spacing) + 1;

      for (let c = 0; c <= cols; c++) {
        const baseX = c * spacing;
        ctx.beginPath();
        for (let r = 0; r <= rows; r++) {
          const p = displace(baseX, r * spacing);
          if (r === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      for (let r = 0; r <= rows; r++) {
        const baseY = r * spacing;
        ctx.beginPath();
        for (let c = 0; c <= cols; c++) {
          const p = displace(c * spacing, baseY);
          if (c === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      ctx.fillStyle = dark
        ? "rgba(251, 251, 249, 0.24)"
        : "rgba(10, 10, 10, 0.18)";
      for (let c = 0; c <= cols; c += 2) {
        for (let r = 0; r <= rows; r += 2) {
          const p = displace(c * spacing, r * spacing);
          ctx.fillRect(p.x - 1, p.y - 1, 2, 2);
        }
      }

      if (pointer.active && pointer.x > -200) {
        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, 44, 0, Math.PI * 2);
        ctx.strokeStyle = dark
          ? "rgba(251, 251, 249, 0.16)"
          : "rgba(10, 10, 10, 0.11)";
        ctx.setLineDash([2, 6]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      rafId = requestAnimationFrame(render);
    };

    rafId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
    />
  );
}

export function OculusMark({
  onTelemetry,
  scrollRatio = 0,
}: {
  onTelemetry?: (t: GazeTelemetry) => void;
  scrollRatio?: number;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const pupilRef = useRef<SVGCircleElement>(null);
  const ringOuterRef = useRef<SVGGElement>(null);
  const ringInnerRef = useRef<SVGGElement>(null);
  const upperBandRef = useRef<SVGPathElement>(null);
  const lowerBandRef = useRef<SVGPathElement>(null);
  const pulseRef = useRef<SVGCircleElement>(null);
  const [calibrating, setCalibrating] = useState(false);

  const stateRef = useRef({
    px: 0,
    py: 0,
    vx: 0,
    vy: 0,
    tx: 0,
    ty: 0,
    rot: 0,
    blink: 1,
    pulseR: 12,
    pulseO: 0,
  });

  useEffect(() => {
    let rafId = 0;
    let lastEmit = 0;
    let nextBlink = performance.now() + 4200;

    const onMove = (e: MouseEvent) => {
      const el = svgRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const nx = (e.clientX - cx) / (window.innerWidth * 0.5);
      const ny = (e.clientY - cy) / (window.innerHeight * 0.5);

      const maxRx = 40;
      const maxRy = 16;
      const angle = Math.atan2(ny, nx);
      const mag = Math.min(Math.hypot(nx, ny), 1);

      stateRef.current.tx = Math.cos(angle) * maxRx * mag;
      stateRef.current.ty = Math.sin(angle) * maxRy * mag;
    };

    window.addEventListener("mousemove", onMove, { passive: true });

    const tick = (now: number) => {
      const s = stateRef.current;
      const stiffness = 0.085;
      const damping = 0.78;

      const ax = (s.tx - s.px) * stiffness;
      const ay = (s.ty - s.py) * stiffness;
      s.vx = (s.vx + ax) * damping;
      s.vy = (s.vy + ay) * damping;
      s.px += s.vx;
      s.py += s.vy;

      const speed = Math.hypot(s.vx, s.vy);
      s.rot += 0.08 + speed * 0.6;

      if (now > nextBlink) {
        s.blink = 0.08;
        nextBlink = now + 3800 + Math.random() * 4200;
      } else {
        s.blink += (1 - s.blink) * 0.18;
      }

      if (s.pulseO > 0.005) {
        s.pulseR += 3.8;
        s.pulseO *= 0.93;
      } else {
        s.pulseO = 0;
      }

      const pupilR = 36.4 + Math.min(speed * 2.2, 5.5);

      if (pupilRef.current) {
        pupilRef.current.setAttribute("cx", (240 + s.px).toFixed(2));
        pupilRef.current.setAttribute("cy", (180 + s.py).toFixed(2));
        pupilRef.current.setAttribute("r", pupilR.toFixed(2));
      }

      if (pulseRef.current) {
        pulseRef.current.setAttribute("cx", (240 + s.px).toFixed(2));
        pulseRef.current.setAttribute("cy", (180 + s.py).toFixed(2));
        pulseRef.current.setAttribute("r", s.pulseR.toFixed(2));
        pulseRef.current.setAttribute("opacity", s.pulseO.toFixed(3));
      }

      if (ringOuterRef.current) {
        ringOuterRef.current.setAttribute(
          "transform",
          `rotate(${s.rot.toFixed(2)} 240 180)`
        );
      }

      if (ringInnerRef.current) {
        ringInnerRef.current.setAttribute(
          "transform",
          `rotate(${(-s.rot * 1.45).toFixed(2)} 240 180)`
        );
      }

      const shear = scrollRatio * 12;
      if (upperBandRef.current) {
        upperBandRef.current.setAttribute(
          "transform",
          `translate(${shear.toFixed(2)} ${(-shear * 0.35).toFixed(2)})`
        );
      }

      if (onTelemetry && now - lastEmit > 60) {
        lastEmit = now;
        onTelemetry({
          yaw: Number(((s.px / 40) * 32).toFixed(1)),
          pitch: Number(((-s.py / 16) * 18).toFixed(1)),
          dilation: Number((pupilR / 36.4).toFixed(2)),
          velocity: Number((speed * 14).toFixed(1)),
        });
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("mousemove", onMove);
    };
  }, [onTelemetry, scrollRatio]);

  const triggerCalibration = () => {
    stateRef.current.pulseR = 36;
    stateRef.current.pulseO = 0.85;
    stateRef.current.blink = 0.1;
    setCalibrating(true);
    setTimeout(() => setCalibrating(false), 650);
  };

  const ticks = Array.from({ length: 72 }, (_, i) => {
    const deg = i * 5;
    const rad = (deg * Math.PI) / 180;
    const major = i % 6 === 0;
    const r1 = major ? 200 : 208;
    const r2 = 216;
    return {
      x1: 240 + Math.cos(rad) * r1,
      y1: 180 + Math.sin(rad) * r1,
      x2: 240 + Math.cos(rad) * r2,
      y2: 180 + Math.sin(rad) * r2,
      major,
    };
  });

  return (
    <div
      onClick={triggerCalibration}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") triggerCalibration();
      }}
      aria-label="Calibrate OCULUS optical aperture"
      className="group relative flex cursor-crosshair select-none items-center justify-center focus:outline-none"
    >
      <svg
        ref={svgRef}
        viewBox="-60 -50 600 460"
        className="h-[250px] w-[300px] overflow-visible text-[var(--oculus-ink)] sm:h-[310px] sm:w-[370px] lg:h-[360px] lg:w-[430px]"
      >
        <g ref={ringOuterRef} opacity="0.28">
          <circle
            cx="240"
            cy="180"
            r="216"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
          />
          {ticks.map((t, i) => (
            <line
              key={i}
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke="currentColor"
              strokeWidth={t.major ? "2" : "1"}
            />
          ))}
        </g>

        <g ref={ringInnerRef} opacity="0.22">
          <circle
            cx="240"
            cy="180"
            r="175"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="6 18"
          />
          <line x1="240" y1="-25" x2="240" y2="5" stroke="currentColor" strokeWidth="1.5" />
          <line x1="240" y1="355" x2="240" y2="385" stroke="currentColor" strokeWidth="1.5" />
          <line x1="-35" y1="180" x2="-5" y2="180" stroke="currentColor" strokeWidth="1.5" />
          <line x1="485" y1="180" x2="515" y2="180" stroke="currentColor" strokeWidth="1.5" />
        </g>

        <g className="oculus-aperture">
          <path
            ref={upperBandRef}
            fillRule="evenodd"
            d="M2,102L1,152L150,66L171,56L192,49L216,44L256,43L293,50L315,58L438,128L439,130L404,151L395,154L305,102L290,95L270,89L249,86L218,88L205,91L182,100L1,204L1,257L129,331L156,344L176,351L215,359L258,360L282,357L311,350L351,333L479,259L479,209L336,292L314,303L295,310L267,316L225,317L192,311L167,302L44,232L44,229L86,205L177,258L200,268L232,274L265,272L287,266L306,257L478,158L479,102L353,29L333,19L309,10L280,3L255,0L215,1L172,10L130,28Z M130,180L209,135L234,129L257,130L281,138L349,177L352,180L274,225L249,231L225,230L205,224L133,183Z"
            fill="currentColor"
            className="transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] group-hover:scale-[1.01]"
          />
          <circle
            ref={pulseRef}
            cx="240"
            cy="180"
            r="36"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            opacity="0"
          />
          <circle ref={pupilRef} cx="240" cy="180" r="36.4" fill="currentColor" />
        </g>
      </svg>

      <div className="pointer-events-none absolute -bottom-2 flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.3em] text-[#525252] transition-opacity duration-300 group-hover:text-[#0A0A0A]">
        <span
          className={`h-1.5 w-1.5 rounded-full transition-colors ${
            calibrating ? "bg-[#E12D1E]" : "bg-[#0A0A0A]"
          }`}
        />
        <span>{calibrating ? "Q0 CALIBRATING" : "CLICK APERTURE TO PULSE"}</span>
      </div>
    </div>
  );
}

export function CipherReadout({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  const [display, setDisplay] = useState(text);
  const elRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const glyphs = "01ΔΦΨΛ/—";
    let frame = 0;
    let rafId = 0;

    const run = () => {
      frame = 0;
      const total = text.length * 2.2;
      const step = () => {
        frame += 1;
        const resolved = Math.floor(frame / 2.2);
        let out = "";
        for (let i = 0; i < text.length; i++) {
          if (text[i] === " " || i < resolved) out += text[i];
          else out += glyphs[(i + frame) % glyphs.length];
        }
        setDisplay(out);
        if (frame < total) rafId = requestAnimationFrame(step);
        else setDisplay(text);
      };
      rafId = requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          run();
          observer.disconnect();
        }
      },
      { threshold: 0.2 }
    );

    if (elRef.current) observer.observe(elRef.current);

    return () => {
      cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, [text]);

  return (
    <span ref={elRef} className={className}>
      {display}
    </span>
  );
}

export function HarmonicWaveform({
  freq = 2.4,
  active = false,
}: {
  freq?: number;
  active?: boolean;
}) {
  const pathRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    let rafId = 0;
    let phase = Math.random() * 10;

    const animate = () => {
      phase += active ? 0.085 : 0.025;
      const points: string[] = [];
      for (let x = 0; x <= 120; x += 3) {
        const env = Math.sin((x / 120) * Math.PI);
        const y =
          14 +
          Math.sin(x * 0.08 * freq + phase) * 8 * env +
          Math.cos(x * 0.19 - phase * 1.4) * 3 * env;
        points.push(`${x === 0 ? "M" : "L"} ${x} ${y.toFixed(1)}`);
      }
      if (pathRef.current) pathRef.current.setAttribute("d", points.join(" "));
      rafId = requestAnimationFrame(animate);
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [freq, active]);

  return (
    <svg viewBox="0 0 120 28" className="h-7 w-28 overflow-visible">
      <line
        x1="0"
        y1="14"
        x2="120"
        y2="14"
        stroke="currentColor"
        strokeOpacity="0.18"
        strokeWidth="0.7"
        strokeDasharray="2 3"
      />
      <path
        ref={pathRef}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.25"
        d="M 0 14 L 120 14"
      />
    </svg>
  );
}

export function IngestionSchematic({
  variant,
}: {
  variant: "demo" | "webrtc" | "bot";
}) {
  if (variant === "demo") {
    return (
      <svg viewBox="0 0 160 64" className="h-14 w-36 overflow-visible">
        <rect
          x="4"
          y="10"
          width="48"
          height="44"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
        />
        <line x1="16" y1="10" x2="16" y2="54" stroke="currentColor" strokeOpacity="0.3" />
        <line x1="40" y1="10" x2="40" y2="54" stroke="currentColor" strokeOpacity="0.3" />
        <path
          d="M 52 32 L 88 32 M 88 18 L 88 46 M 88 18 L 118 18 M 88 46 L 118 46"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
        />
        <rect x="118" y="11" width="36" height="14" fill="none" stroke="currentColor" strokeWidth="1" />
        <rect x="118" y="39" width="36" height="14" fill="none" stroke="currentColor" strokeWidth="1" />
        <circle cx="88" cy="32" r="2.5" fill="currentColor" />
        <line
          x1="28"
          y1="14"
          x2="28"
          y2="50"
          stroke="currentColor"
          strokeWidth="1.2"
          className="origin-center transition-transform duration-700 group-hover:translate-x-2"
        />
      </svg>
    );
  }

  if (variant === "webrtc") {
    return (
      <svg viewBox="0 0 160 64" className="h-14 w-36 overflow-visible">
        <circle cx="28" cy="32" r="14" fill="none" stroke="currentColor" strokeWidth="1" />
        <circle cx="28" cy="32" r="3" fill="currentColor" />
        <circle cx="132" cy="32" r="14" fill="none" stroke="currentColor" strokeWidth="1" />
        <circle cx="132" cy="32" r="3" fill="currentColor" />
        <path
          d="M 42 26 Q 80 8, 118 26"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="3 3"
          className="transition-all duration-700 group-hover:stroke-dashoffset-12"
        />
        <path
          d="M 118 38 Q 80 56, 42 38"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
        />
        <circle cx="80" cy="32" r="5" fill="none" stroke="currentColor" strokeWidth="1" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 160 64" className="h-14 w-36 overflow-visible">
      <circle
        cx="80"
        cy="32"
        r="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="0.9"
        strokeDasharray="2 4"
        className="origin-[80px_32px] transition-transform duration-1000 group-hover:rotate-45"
      />
      <path
        d="M 54 32 Q 80 14, 106 32 Q 80 50, 54 32 Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <circle cx="80" cy="32" r="4.5" fill="currentColor" />
      <line x1="16" y1="32" x2="44" y2="32" stroke="currentColor" strokeWidth="0.9" />
      <line x1="116" y1="32" x2="144" y2="32" stroke="currentColor" strokeWidth="0.9" />
    </svg>
  );
}
