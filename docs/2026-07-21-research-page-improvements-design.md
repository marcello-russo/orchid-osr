# Research Page Improvements — Design Doc

## Objective

Update dataset numbers to real counts, fix TOC sidebar padding, add citation popover anchored to superscript.

## Changes

### 1. Dataset numbers — aggiornare conteggi

Nel file `frontend/src/pages/ResearchPage.tsx`:

**Linea 159:** `"2.880 immagini"` → `"3.040 immagini"`

**Tabella distribuzione (linee 229-236):**

| Specie | Train (vecchio) | Train (nuovo) | Test |
|--------|:--------------:|:-------------:|:----:|
| O. exaltata | 501 | 490 | 50 |
| O. garganica | 570 | 563 | 50 |
| O. incubacea | 366 | 352 | 50 |
| O. majellensis | 152 | 152 | 50 |
| O. sphegodes | 433 | 425 | 50 |
| O. sphegodes Palena | 210 | 210 | 50 |
| O. sphegodes_Napoli (novel) | — | 0 | 548 |
| **Totale** | 2.232 | **2.192** | **848** |

### 2. Sidebar TOC — padding sinistro

`frontend/src/pages/ResearchPage.tsx` linea 96:

```tsx
// Prima:
<aside className="fixed left-0 top-24 bottom-0 w-56 hidden xl:block overflow-y-auto border-r border-border/50 pr-4">

// Dopo:
<aside className="fixed left-0 top-24 bottom-0 w-64 hidden xl:block overflow-y-auto border-r border-border/50 pr-4 pl-6">
```

- `w-56` → `w-64` (compensa l'extra padding)
- Aggiunto `pl-6`

### 3. Citation Popover

**Componenti coinvolti:**
- `frontend/src/features/research/components/Citation.tsx` — riscritto
- `frontend/src/pages/ResearchPage.tsx` — aggiunto container per il popover

**Come funziona:**

1. Il `<Cite>` è un `<span className="relative inline-block">`. Al click imposta l'ID attivo via context.
2. Il popover è renderizzato dentro lo `<span>` del `Cite`, con `position: fixed`.
3. Quando si apre, calcola `top = superscriptRect.bottom + 4` e `left = contentRect.left` dal `getBoundingClientRect()`.
4. Larghezza: fissa uguale alla larghezza del container del contenuto (`contentRect.width`).
5. Un scroll listener aggiorna la posizione a ogni scroll — il popover segue il superscript.
6. Chiusura: click fuori, Escape, o se si clicca un altro `Cite`.
7. Lo scroll della pagina non è bloccato.

**Perché `fixed` e non `absolute`:** Il popover deve essere renderizzato dentro il `<span>` del `Cite` per scrollare col contenuto. Ma con `absolute` dentro `<span>`, la larghezza `left-0 right-0` sarebbe relativa allo `<span>` (stretta). Con `fixed`, `left` e `width` sono calcolati sul viewport, e lo scroll listener mantiene l'ancoraggio.

**Dettaglio implementativo — componente Cite riscritto:**

Il `<Cite>` usa un context (`useCiteContext`) per condividere `activeCite` e `setActiveCite`. Il popover è renderizzato dentro lo stesso `<span>`, usa `position: fixed`, e aggiorna la posizione via scroll listener.

```tsx
export function Cite({ id }: { id: string }) {
  const { activeCite, setActiveCite } = useCiteContext();
  const paper = PAPER_MAP[id];
  const isOpen = activeCite === id;

  if (!paper) return <sup className="text-muted-foreground text-[10px]">[{id}]</sup>;

  return (
    <span className="relative inline-block">
      <sup
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

```tsx
// CitationPopover — fixed positioning con scroll tracking
function CitationPopover({ paper, onClose }: { paper: Paper; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 800 });
  const supRef = useRef<HTMLElement | null>(null);

  // Calcola posizione iniziale e aggiorna a ogni scroll
  const updatePos = useCallback(() => {
    if (!supRef.current) return;
    const sup = supRef.current.getBoundingClientRect();
    const container = document.querySelector(".research-content") as HTMLElement;
    if (!container) return;
    const cont = container.getBoundingClientRect();
    setPos({ top: sup.bottom + 4, left: cont.left, width: cont.width });
  }, []);

  useEffect(() => {
    supRef.current = document.getElementById(`cite-${paper.id}`);
    updatePos();
    window.addEventListener("scroll", updatePos, { passive: true });
    return () => window.removeEventListener("scroll", updatePos);
  }, [updatePos, paper.id]);

  // Click fuori
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  // Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      style={{ position: "fixed", top: pos.top, left: pos.left, width: pos.width }}
      className="bg-card border border-border rounded-lg p-4 shadow-lg z-50 animate-in fade-in duration-200"
    >
      <p className="text-sm font-medium">{paper.authors} ({paper.year})</p>
      <p className="text-sm text-foreground/80 italic mt-1">{paper.title}</p>
      {paper.venue && <p className="text-xs text-muted-foreground mt-1">{paper.venue}</p>}
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

Il superscript nel DOM deve avere un `id="cite-{paper.id}"` per permettere al popover di trovarlo. Questo va aggiunto al `<sup>` nel `Cite`:

```tsx
<sup
  id={`cite-${paper.id}`}
  className="..."
>
```

**Container research-content:** Aggiungere `className="research-content relative"` al `<div>` principale del contenuto in `ResearchPage.tsx` per fungere da ancoraggio per il calcolo della larghezza.

**Stato condiviso:** Un context React (`CiteContext`) fornisce `activeCite: string | null` e `setActiveCite`. Il `ResearchPage` wrappa il contenuto in `<CiteProvider>`. I `Cite` leggono e scrivono lo stato. Il popover è renderizzato dentro `Cite.tsx`.

**Layout parent:** Il `<div>` principale del contenuto in `ResearchPage.tsx` deve avere `className="research-content relative"` per fungere da ancoraggio per il calcolo della larghezza del popover (`querySelector(".research-content")`).

## Files modificati

| File | Modifica |
|------|----------|
| `frontend/src/pages/ResearchPage.tsx` | Numeri dataset + padding TOC + `relative` al container |
| `frontend/src/features/research/components/Citation.tsx` | Riscritto: click popover invece di link diretto |

## Test

- Click su una citazione → appare il popover sotto il numeretto
- Click fuori → popover si chiude
- Escape → popover si chiude
- Click sul link "Apri su ↗" → apre il paper in nuova tab
- Scroll pagina → popover scrolla col contenuto
- Mobile → popover full width
- Tabella distribuzione → mostra 7 specie con conteggi corretti
- TOC sidebar → padding sinistro aumentato
