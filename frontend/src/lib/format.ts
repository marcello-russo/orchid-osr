// frontend/src/lib/format.ts
//
// Formattazione numerica legata alla lingua dell'app, non al browser.
// `toFixed()` produce sempre il punto: in italiano "83.00%" invece di "83,00%",
// mentre la prosa della tesi scrive "0,833". Questi helper chiudono la
// discrepanza e seguono il cambio lingua senza ricaricare.

// La lingua si legge da `<html lang>`, che `lib/i18n.ts` aggiorna a ogni cambio:
// così questo modulo non importa l'istanza i18n (che nei test è mockata) e non
// crea cicli.
function locale(lang?: string): string {
  if (lang) return lang;
  if (typeof document !== "undefined" && document.documentElement.lang) {
    return document.documentElement.lang;
  }
  return "it";
}

/** Numero con separatori e cifre decimali della lingua corrente. */
export function formatNumber(value: number, digits = 2, lang?: string): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale(lang), {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

/**
 * Percentuale da una frazione (0.833 -> "83,30%" in IT, "83.30%" in EN).
 * Accetta anche valori fuori da [0,1] (es. rapporti > 1): li formatta come
 * percentuale lo stesso, come faceva `value * 100`.
 */
export function formatPercent(value: number, digits = 2, lang?: string): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale(lang), {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

/** Data (e ora) nella lingua dell'app. */
export function formatDate(iso: string, lang?: string, withTime = false): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(locale(lang), withTime ? undefined : { dateStyle: "medium" });
}
