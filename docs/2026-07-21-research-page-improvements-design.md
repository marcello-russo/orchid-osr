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

1. Il `<Cite>` diventa cliccabile (non più un link diretto). Al click imposta uno stato globale (o un context) con l'ID della citazione.
2. Un componente `<CitationPopover>` viene renderizzato alla fine del main content (dentro il container `max-w-7xl` che ha `position: relative`).
3. Quando un ID è attivo, il popover:
   - Si posiziona con `position: absolute` usando le coordinate del superscript (calcolate via `getBoundingClientRect`)
   - Ha `width: 100%` del container parent
   - Mostra: autori, anno, titolo, venue, e link "Apri su ↗"
   - Ha un bordo e sfondo che lo distinguono dal contenuto
4. Chiusura: click fuori dal popover o tasto Escape.
5. Lo scroll della pagina non è bloccato — il popover scrolla col contenuto.

**Dettaglio implementativo:**

```tsx
// CitationPopover (nuovo componente)
function CitationPopover({ paper, onClose }: { paper: Paper; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute left-0 right-0 bg-card border border-border rounded-lg p-4 shadow-lg z-10 animate-in fade-in slide-in-from-top-2 duration-200"
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

**Stato condiviso:** Un contesto React o una semplice prop drilling. Il `ResearchPage` mantiene uno stato `activeCite: string | null` e lo passa al `Cite` e al `CitationPopover`.

**Layout parent:** Il container principale `max-w-7xl` (o il suo wrapper) deve avere `position: relative` per far funzionare il `absolute` del popover.

## Files modificati

| File | Modifica |
|------|----------|
| `frontend/src/pages/ResearchPage.tsx` | Numeri dataset, padding TOC, stato activeCite + CitationPopover |
| `frontend/src/features/research/components/Citation.tsx` | Riscritto: click → popover invece di link diretto |

## Test

- Click su una citazione → appare il popover sotto il numeretto
- Click fuori → popover si chiude
- Escape → popover si chiude
- Click sul link "Apri su ↗" → apre il paper in nuova tab
- Scroll pagina → popover scrolla col contenuto
- Mobile → popover full width
- Tabella distribuzione → mostra 7 specie con conteggi corretti
- TOC sidebar → padding sinistro aumentato
