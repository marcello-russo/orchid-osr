import { useRef, useEffect, useState, useCallback } from "react";

interface Point2D {
  x: number; y: number; species: string; idx: number;
}

interface Scatter2DProps {
  points: Point2D[];
  speciesColors: Record<string, string>;
  onSelect?: (idx: number | null) => void;
}

const PALETTE = [
  "#3B82F6", "#D97706", "#10B981", "#8B5CF6",
  "#EF4444", "#EC4899", "#14B8A6", "#F97316",
];

export function Scatter2D({ points, speciesColors, onSelect }: Scatter2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 600, h: 400 });
  const sizeRef = useRef({ w: 600, h: 400 });
  const panRef = useRef({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const dragRef = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<Point2D | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const bounds = useCallback(() => {
    if (points.length === 0) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const pad = Math.max((maxX - minX) * 0.05, (maxY - minY) * 0.05, 0.1);
    return { minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad };
  }, [points]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;
    const ctx = canvas.getContext("2d")!;
    const s = sizeRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = s.w * dpr; canvas.height = s.h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, s.w, s.h);

    const b = bounds();
    const scaleX = (s.w * 0.85) / (b.maxX - b.minX) * zoomRef.current;
    const scaleY = (s.h * 0.85) / (b.maxY - b.minY) * zoomRef.current;
    const scale = Math.min(scaleX, scaleY);
    const cx = s.w / 2 + panRef.current.x;
    const cy = s.h / 2 + panRef.current.y;

    const toScreen = (px: number, py: number) => ({
      sx: cx + (px - (b.minX + b.maxX) / 2) * scale,
      sy: cy + (py - (b.minY + b.maxY) / 2) * scale,
    });

    const pointDensity = points.length / (s.w * s.h);
    const r = Math.max(1.5, Math.min(4, 3 - pointDensity * 5000));

    for (const p of points) {
      const { sx, sy } = toScreen(p.x, p.y);
      if (sx < -10 || sx > s.w + 10 || sy < -10 || sy > s.h + 10) continue;
      const color = speciesColors[p.species] || PALETTE[0];
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      if (p === hovered) {
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [points, speciesColors, bounds, hovered]);

  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => { draw(); }, [points, speciesColors]);
  useEffect(() => {
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0 && height > 0) { sizeRef.current = { w: Math.floor(width), h: Math.floor(height) }; setSize(sizeRef.current); drawRef.current(); }
      }
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const findPoint = useCallback((mx: number, my: number) => {
    const b = bounds();
    const s = sizeRef.current;
    const scaleX = (s.w * 0.85) / (b.maxX - b.minX) * zoomRef.current;
    const scaleY = (s.h * 0.85) / (b.maxY - b.minY) * zoomRef.current;
    const scale = Math.min(scaleX, scaleY);
    const cx = s.w / 2 + panRef.current.x;
    const cy = s.h / 2 + panRef.current.y;
    const pointDensity = points.length / (s.w * s.h);
    const r = Math.max(1.5, Math.min(4, 3 - pointDensity * 5000));
    const threshold = Math.max(10, r * 3);

    for (const p of points) {
      const sx = cx + (p.x - (b.minX + b.maxX) / 2) * scale;
      const sy = cy + (p.y - (b.minY + b.maxY) / 2) * scale;
      if (Math.abs(mx - sx) < threshold && Math.abs(my - sy) < threshold) return p;
    }
    return null;
  }, [points, bounds, size]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragRef.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (dragRef.current) {
      panRef.current.x += e.clientX - lastPos.current.x;
      panRef.current.y += e.clientY - lastPos.current.y;
      lastPos.current = { x: e.clientX, y: e.clientY };
      draw(); return;
    }
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const p = findPoint(e.clientX - rect.left, e.clientY - rect.top);
    setHovered(p);
    (e.target as HTMLElement).style.cursor = p ? "pointer" : "grab";
    setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top - 10 });
  }, [findPoint, draw]);

  const onPointerUp = useCallback(() => { dragRef.current = false; }, []);
  const onClick = useCallback(() => {
    if (hovered) onSelect?.(hovered.idx);
  }, [hovered, onSelect]);
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    zoomRef.current *= 1 - e.deltaY * 0.001;
    zoomRef.current = Math.max(0.3, Math.min(10, zoomRef.current));
    draw();
  }, [draw]);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[400px]">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing rounded-xl"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onClick={onClick}
        onWheel={onWheel}
        style={{ touchAction: "none" }}
      />
      {hovered && (
        <div className="absolute pointer-events-none bg-background/90 backdrop-blur-sm border border-border/50 rounded-lg px-2.5 py-1.5 text-xs shadow-lg z-10"
          style={{ left: tooltipPos.x, top: tooltipPos.y, transform: "translate(-50%, -100%)" }}>
          <span className="inline-block size-2 rounded-full mr-1.5 align-middle"
            style={{ backgroundColor: speciesColors[hovered.species] || PALETTE[0] }} />
          <span className="font-medium">{hovered.species}</span>
        </div>
      )}
    </div>
  );
}
