// frontend/src/features/research/hooks/useActiveSection.ts

import { useEffect, useState } from "react";

/** Sezioni della pagina di ricerca: ordine della TOC e ancore dei link. */
export const SECTIONS = [
  { id: "intro", labelKey: "research.sect_intro" },
  { id: "contesto", labelKey: "research.sect_contesto" },
  { id: "dataset", labelKey: "research.sect_dataset" },
  { id: "metodologia", labelKey: "research.sect_metodologia" },
  { id: "exp1", labelKey: "research.sect_exp1" },
  { id: "exp2", labelKey: "research.sect_exp2" },
  { id: "exp3", labelKey: "research.sect_exp3" },
  { id: "exp4", labelKey: "research.sect_exp4" },
  { id: "exp5", labelKey: "research.sect_exp5" },
  { id: "benchmark", labelKey: "research.sect_benchmark" },
  { id: "conclusioni", labelKey: "research.sect_conclusioni" },
  { id: "bibliografia", labelKey: "research.sect_bibliografia" },
];

/** Id della sezione visibile, per evidenziare la voce corrispondente nella TOC. */
export function useActiveSection() {
  const [active, setActive] = useState("");
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) { setActive(e.target.id); break; }
        }
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );
    for (const { id } of SECTIONS) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);
  return active;
}
