import { useEffect, useState, useRef, useCallback, useLayoutEffect } from "react";
import { useTranslation } from "react-i18next";
import type { Paper } from "@/features/research/data/papers";

interface CitationPopoverProps {
  paper: Paper;
  onClose: () => void;
}

export function CitationPopover({ paper, onClose }: CitationPopoverProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  const updatePos = useCallback(() => {
    const sup = document.getElementById(`cite-${paper.id}`);
    if (!sup) return;
    const supRect = sup.getBoundingClientRect();
    const popoverHeight = ref.current?.offsetHeight ?? 0;
    const POPOVER_WIDTH = 448; // max-w-md (28rem)
    const margin = 8;
    const left = Math.min(
      Math.max(margin, supRect.left - margin),
      window.innerWidth - POPOVER_WIDTH - margin,
    );
    let top = supRect.bottom + margin;
    if (top + popoverHeight > window.innerHeight - margin) {
      top = Math.max(margin, supRect.top - popoverHeight - margin);
    }
    setPos({ top, left });
  }, [paper.id]);

  const measured = useRef(false);

  useLayoutEffect(() => {
    updatePos();
  }, [updatePos]);

  useEffect(() => {
    if (!pos || measured.current) return;
    measured.current = true;
    // Re-position now that the popover is mounted and its real height is known.
    updatePos();
  }, [pos, updatePos]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
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

  if (!pos) return null;

  return (
    <div
      ref={ref}
      style={{ position: "fixed", top: pos.top, left: pos.left }}
      className="bg-card border border-border rounded-lg p-4 shadow-lg z-50 animate-in fade-in duration-200 w-auto max-w-md"
    >
      <p className="text-sm font-medium">
        {paper.authors}
      </p>
      <p className="text-sm text-foreground/80 mt-0.5">({paper.year})</p>
      <p className="text-sm text-foreground/80 italic mt-1">{paper.title}</p>
      {paper.venue && (
        <p className="text-xs text-muted-foreground mt-1">{paper.venue}</p>
      )}
      <a
        href={paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : "#")}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-2"
      >
        {t("research.cite_open")}
      </a>
    </div>
  );
}
