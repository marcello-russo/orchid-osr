# Research Page Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update dataset numbers, fix TOC sidebar padding, add citation popover with scroll-tracking.

**Architecture:** New `CiteContext` for shared popover state, `CitationPopover` component with `position:fixed` + scroll listener, `Citation.tsx` rewritten to trigger popover on click. ResearchPage updated with new numbers + CSS classes.

**Tech Stack:** React + TypeScript + Tailwind

---

### Task 1: Dataset numbers — aggiornare conteggi

**Files:**
- Modify: `frontend/src/pages/ResearchPage.tsx`

- [ ] **Step 1: Aggiornare "2.880 immagini" in "3.040 immagini"**

Trova la stringa `2.880 immagini` e sostituiscila con `3.040 immagini`.

Prima:
```tsx
<li>Dataset curato di 2.880 immagini con split train/test privi di data leakage</li>
```
Dopo:
```tsx
<li>Dataset curato di 3.040 immagini con split train/test privi di data leakage</li>
```

- [ ] **Step 2: Aggiornare la tabella distribuzione**

Trova il blocco `rows` del DataTable (dove ci sono `train: "501"`, `train: "570"`, ecc.).

Sostituire INTERAMENTE il blocco `rows` da `rows={[` fino a `]}` con:

```tsx
rows={[
  { species: "O. exaltata", train: "490", test: "50" },
  { species: "O. garganica", train: "563", test: "50" },
  { species: "O. incubacea", train: "352", test: "50" },
  { species: "O. majellensis", train: "152", test: "50" },
  { species: "O. sphegodes", train: "425", test: "50" },
  { species: "O. sphegodes Palena", train: "210", test: "50" },
  { species: "O. sphegodes_Napoli (novel)", train: "0", test: "548" },
  { species: "Totale", train: "2.192", test: "848" },
]}
```

- [ ] **Step 3: Aggiornare la caption della tabella**

Prima (caption attuale):
```tsx
caption="Distribuzione delle immagini per specie negli split di training e test."
```
Dopo:
```tsx
caption="Distribuzione delle immagini per specie negli split di training e test. La specie novel è presente solo nel test set (548 immagini)."
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/ResearchPage.tsx
git commit -m "fix: aggiornati numeri dataset ResearchPage (3.040, 7 specie, 848 test)"
```

---

### Task 2: TOC sidebar — padding sinistro

**Files:**
- Modify: `frontend/src/pages/ResearchPage.tsx`

- [ ] **Step 1: Modificare le classi dell'aside TOC**

Trova l'elemento `<aside className="fixed left-0 top-24 bottom-0 w-56 ...">`.

Modificare `w-56` in `w-64` e aggiungere `pl-6`:

Prima:
```tsx
<aside className="fixed left-0 top-24 bottom-0 w-56 hidden xl:block overflow-y-auto border-r border-border/50 pr-4">
```
Dopo:
```tsx
<aside className="fixed left-0 top-24 bottom-0 w-64 hidden xl:block overflow-y-auto border-r border-border/50 pr-4 pl-6">
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/ResearchPage.tsx
git commit -m "fix: TOC sidebar padding sinistro (w-64 pl-6)"
```

---

### Task 3: CiteContext — stato condiviso per popover

**Files:**
- Create: `frontend/src/features/research/components/CiteContext.tsx`

- [ ] **Step 1: Creare il file CiteContext.tsx**

Creare il file `frontend/src/features/research/components/CiteContext.tsx` con:

```tsx
import { createContext, useContext, useState, type ReactNode } from "react";

interface CiteContextValue {
  activeCite: string | null;
  setActiveCite: (id: string | null) => void;
}

const CiteContext = createContext<CiteContextValue | null>(null);

export function CiteProvider({ children }: { children: ReactNode }) {
  const [activeCite, setActiveCite] = useState<string | null>(null);
  return (
    <CiteContext.Provider value={{ activeCite, setActiveCite }}>
      {children}
    </CiteContext.Provider>
  );
}

export function useCiteContext(): CiteContextValue {
  const ctx = useContext(CiteContext);
  if (!ctx) throw new Error("useCiteContext must be used inside CiteProvider");
  return ctx;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/features/research/components/CiteContext.tsx
git commit -m "feat: CiteContext per stato popover citazioni"
```

---

### Task 4: CitationPopover — componente popover fisso

**Files:**
- Create: `frontend/src/features/research/components/CitationPopover.tsx`

- [ ] **Step 1: Creare CitationPopover.tsx**

```tsx
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
    const onMouseDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [onClose]);

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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/features/research/components/CitationPopover.tsx
git commit -m "feat: CitationPopover con fixed positioning e scroll tracking"
```

---

### Task 5: Cite — riscrivere con context + popover

**Files:**
- Modify: `frontend/src/features/research/components/Citation.tsx`

- [ ] **Step 1: Riscrivere Citation.tsx**

Sostituire TUTTO il contenuto di `frontend/src/features/research/components/Citation.tsx` con:

```tsx
import { useCiteContext } from "@/features/research/components/CiteContext";
import { CitationPopover } from "@/features/research/components/CitationPopover";
import { PAPER_MAP } from "@/features/research/data/papers";

export function Cite({ id }: { id: string }) {
  const { activeCite, setActiveCite } = useCiteContext();
  const paper = PAPER_MAP[id];
  const isOpen = activeCite === id;

  if (!paper) {
    return <sup className="text-muted-foreground text-[10px]">[{id}]</sup>;
  }

  return (
    <span className="relative inline-block">
      <sup
        id={`cite-${paper.id}`}
        className="text-primary text-[10px] font-medium cursor-pointer hover:underline"
        onClick={() => setActiveCite(isOpen ? null : id)}
      >
        [{paper.authors}, {paper.year}]
      </sup>
      {isOpen && <CitationPopover paper={paper} onClose={() => setActiveCite(null)} />}
    </span>
  );
}
```

Note: rimuovere `import { PAPER_MAP } from "@/features/research/data/papers"` dall'alto se era già importato — ora è nell'import sopra.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/features/research/components/Citation.tsx
git commit -m "feat: Cite con popover (context + fixed positioning)"
```

---

### Task 6: ResearchPage — integrare context e classi

**Files:**
- Modify: `frontend/src/pages/ResearchPage.tsx`

- [ ] **Step 1: Aggiungere l'import del CiteProvider**

In cima a `ResearchPage.tsx`, tra gli import esistenti, aggiungere:

```tsx
import { CiteProvider } from "@/features/research/components/CiteContext";
```

- [ ] **Step 2: Aggiungere className al container principale**

Trovare il `<div>` principale del contenuto (quello con `max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700`). Aggiungere `research-content relative` al className:

Prima:
```tsx
<div className="space-y-8 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
```
Dopo:
```tsx
<div className="research-content relative space-y-8 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
```

Nota: `space-y-8` potrebbe essere su un div DIVERSO dal `max-w-7xl`. Cerca il div con `max-w-7xl` e aggiungi `research-content relative` ALLO STESSO div.

- [ ] **Step 3: Wrappare il contenuto in CiteProvider**

Trovare il punto in cui viene renderizzato il contenuto principale (il `<div className="research-content relative ...">` o il suo genitore). Wrappare TUTTO il contenuto della pagina (dopo la hero section) in `<CiteProvider>`.

Se la struttura è:
```tsx
return (
  <div ...> // research-content
    <Section id="intro">...</Section>
    ...
  </div>
);
```

Aggiungere:
```tsx
return (
  <CiteProvider>
    <div className="research-content relative ...">
      <Section id="intro">...</Section>
      ...
    </div>
  </CiteProvider>
);
```

Nota: il `CiteProvider` deve wrappare TUTTI i `<Cite>` usati nella pagina. Assicurati che sia il genitore più esterno possibile (dopo i componenti layout che non usano cite).

- [ ] **Step 4: Build di verifica**

```bash
cd frontend && npx tsc --noEmit
```

Deve passare senza errori.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/ResearchPage.tsx
git commit -m "feat: ResearchPage integra CiteProvider + research-content class"
```

---

### Task 7: Verifica build finale

- [ ] **Step 1: Build frontend**

```bash
cd frontend && npm run build
```

Nessun errore TypeScript o Vite.

- [ ] **Step 2: Push e CI**

```bash
git push origin main
```

CI esegue backend test → docker → frontend. Se tutto passa, deploy automatico in VPS.

---

### Riepilogo file per task

| Task | File | Modifica |
|------|------|----------|
| 1 | `frontend/src/pages/ResearchPage.tsx` | Numeri dataset |
| 2 | `frontend/src/pages/ResearchPage.tsx` | TOC padding |
| 3 | `frontend/src/features/research/components/CiteContext.tsx` | **Nuovo** |
| 4 | `frontend/src/features/research/components/CitationPopover.tsx` | **Nuovo** |
| 5 | `frontend/src/features/research/components/Citation.tsx` | Riscritto |
| 6 | `frontend/src/pages/ResearchPage.tsx` | CiteProvider + classi |
