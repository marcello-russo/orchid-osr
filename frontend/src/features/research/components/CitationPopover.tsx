import { useEffect, useState, useRef, useCallback } from "react";
import type { Paper } from "@/features/research/data/papers";

interface CitationPopoverProps {
  paper: Paper;
  onClose: () => void;
}

export function CitationPopover({ paper, onClose }: CitationPopoverProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 800 });

  const updatePos = useCallback(() => {
    const sup = document.getElementById(`cite-${paper.id}`);
    const container = document.querySelector(".research-content");
    if (!sup || !container) return;
    const supRect = sup.getBoundingClientRect();
    const contRect = container.getBoundingClientRect();
    setPos({
      top: supRect.bottom + 4,
      left: contRect.left,
      width: contRect.width,
    });
  }, [paper.id]);

  useEffect(() => {
    updatePos();
    window.addEventListener("scroll", updatePos, { passive: true });
    return () => window.removeEventListener("scroll", updatePos);
  }, [updatePos]);

  useEffect(() => {
    const sup = document.getElementById(`cite-${paper.id}`);
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current && !ref.current.contains(target) && target !== sup) {
        onClose();
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [onClose, paper.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (pos.top === 0) return null;

  return (
    <div
      ref={ref}
      style={{ position: "fixed", top: pos.top, left: pos.left, width: pos.width }}
      className="bg-card border border-border rounded-lg p-4 shadow-lg z-50 animate-in fade-in duration-200"
    >
      <p className="text-sm font-medium">
        {paper.authors} ({paper.year})
      </p>
      <p className="text-sm text-foreground/80 italic mt-1">{paper.title}</p>
      {paper.venue && (
        <p className="text-xs text-muted-foreground mt-1">{paper.venue}</p>
      )}
      <a
        href={paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : "#")}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-2"
        onClick={(e) => e.stopPropagation()}
      >
        Apri su ↗
      </a>
    </div>
  );
}
