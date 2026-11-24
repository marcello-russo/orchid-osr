import React, { useRef, useState, useMemo, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { speciesLabel } from "@/lib/format";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

interface Point3D {
  x: number; y: number; z: number;
  species: string;
  idx: number;
}

interface Scatter3DProps {
  points: Point3D[];
  species: string[];
  speciesColors: Record<string, string>;
  onSelect?: (idx: number | null) => void;
}

const PALETTE = [
  "#3B82F6", "#D97706", "#10B981", "#8B5CF6",
  "#EF4444", "#EC4899", "#14B8A6", "#F97316",
];

function hexToColor(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

function makeCircleTexture(sz = 64): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = sz;
  canvas.height = sz;
  const ctx = canvas.getContext("2d")!;
  ctx.beginPath();
  ctx.arc(sz / 2, sz / 2, sz / 2 - 2, 0, Math.PI * 2);
  ctx.fillStyle = "white";
  ctx.fill();
  return new THREE.CanvasTexture(canvas);
}

const CIRCLE_MAP = makeCircleTexture();

const PointsMesh = React.memo(function PointsMesh({
  points,
  speciesColors,
  setHoveredIdx,
}: {
  points: Point3D[];
  speciesColors: Record<string, string>;
  setHoveredIdx: (idx: number | null) => void;
}) {
  const ref = useRef<THREE.Points>(null!);

  const positions = useMemo(() => {
    const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
    const cz = points.reduce((s, p) => s + p.z, 0) / points.length;
    const pos = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      pos[i * 3] = points[i].x - cx;
      pos[i * 3 + 1] = points[i].y - cy;
      pos[i * 3 + 2] = points[i].z - cz;
    }
    return pos;
  }, [points]);

  const colors = useMemo(() => {
    const col = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      const c = hexToColor(speciesColors[points[i].species] || PALETTE[0]);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    return col;
  }, [points, speciesColors]);

  const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    try {
      if (e.intersections?.[0]?.index !== undefined) {
        setHoveredIdx(e.intersections[0].index);
        const el = e.eventObject?.parent?.parent as HTMLElement | null;
        if (el?.style) el.style.cursor = "pointer";
      }
    } catch { /* three.js rAF errors */ }
  }, [setHoveredIdx]);

  const handlePointerOut = useCallback(() => {
    setHoveredIdx(null);
    if (document.querySelector("canvas")) {
      (document.querySelector("canvas") as HTMLElement).style.cursor = "grab";
    }
  }, [setHoveredIdx]);

  return (
    <points
      ref={ref}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      onClick={(e) => {
        const idx = e.intersections[0]?.index;
        if (idx !== undefined) {
          document.dispatchEvent(new CustomEvent("point-select", { detail: { idx } }));
        }
      }}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={1.5} vertexColors transparent opacity={0.85} sizeAttenuation depthWrite={false} map={CIRCLE_MAP} />
    </points>
  );
});

function HoverGlow({ point }: { point: Point3D | null }) {
  if (!point) return null;
  return (
    <sprite position={[point.x, point.y, point.z]} scale={[0.8, 0.8, 1]}>
      <spriteMaterial map={CIRCLE_MAP} color="white" transparent opacity={0.5} depthWrite={false} />
    </sprite>
  );
}

function Tooltip({ point, speciesColors }: {
  point: Point3D | null;
  speciesColors: Record<string, string>;
}) {
  if (!point) return null;
  return (
    <Html position={[point.x, point.y + 0.06, point.z]} center style={{ pointerEvents: "none" }}>
      <div className="bg-background/90 backdrop-blur-sm border border-border/50 rounded-lg px-2.5 py-1.5 text-xs shadow-lg whitespace-nowrap">
        <span
          className="inline-block size-2 rounded-full mr-1.5 align-middle"
          style={{ backgroundColor: speciesColors[point.species] || PALETTE[0] }}
        />
        <span className="font-medium">{speciesLabel(point.species)}</span>
      </div>
    </Html>
  );
}

export function Scatter3D({ points, speciesColors, onSelect }: Scatter3DProps) {
  const { t } = useTranslation();
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const hoveredPoint = useMemo(
    () => (hoveredIdx !== null ? points[hoveredIdx] ?? null : null),
    [hoveredIdx, points],
  );

  const bounds = useMemo(() => {
    if (points.length === 0) return { cx: 0, cy: 0, cz: 0, scale: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    }
    return {
      cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, cz: (minZ + maxZ) / 2,
      scale: Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.01),
    };
  }, [points]);

  useEffect(() => {
    const handler = (e: Event) => {
      const { idx } = (e as CustomEvent).detail;
      onSelect?.(idx);
    };
    document.addEventListener("point-select", handler);
    return () => document.removeEventListener("point-select", handler);
  }, [onSelect]);

  if (points.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        {t("research.scatter_no_points")}
      </div>
    );
  }

  const dist = bounds.scale * 1.2;

  return (
    <div className="w-full h-full rounded-xl overflow-hidden">
      <Canvas
        camera={{ position: [dist * 0.5, dist * 0.3, dist], fov: 50, near: 0.001, far: dist * 5 }}
        gl={{ antialias: true }}
        style={{ background: "hsl(var(--background))" }}
      >
        <PointsMesh
          points={points}
          speciesColors={speciesColors}
          setHoveredIdx={setHoveredIdx}
        />
        <HoverGlow point={hoveredPoint} />
        <Tooltip point={hoveredPoint} speciesColors={speciesColors} />
        <gridHelper args={[dist * 2, 20]} />
        <OrbitControls enablePan enableZoom enableRotate minDistance={dist * 0.1} maxDistance={dist * 5} />
      </Canvas>
    </div>
  );
}
