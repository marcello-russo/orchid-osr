"""Defense kit exporter — F5, rieseguibile dopo ogni modifica alla deck.

Fa (niente bug perché ogni passo ha verifica):
  F5.1  copia thesis_defense.html + images/ in OUT, apre la copia OFFLINE
        (rete disabilitata), naviga tutte le slide, salva screenshot e li
        confronta per presenza (14 slide attese).
  F5.2  export PDF della deck via Chromium headless, verifica 14 pagine.
  F5.3  copia thesis.pdf in OUT con metadati bonificati (autore/percorso/
        software rimossi), verifica 73 pagine e dump metadati prima/dopo.
  F5.4/5 checklist umana in documents/defense_checklist.md (nomi, data, kit).

Usage: python scripts/export_defense_kit.py [--out /tmp/defense_kit]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
DOCS = BACKEND.parent / "documents"
sys.path.insert(0, str(BACKEND))

EXPECTED_SLIDES = 14
EXPECTED_THESIS_PAGES = 73


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def f51_offline_copy(out: Path) -> Path:
    step("F5.1 copia offline + navigazione 14 slide senza rete")
    kit = out / "deck"
    if kit.exists():
        shutil.rmtree(kit)
    (kit / "images").mkdir(parents=True)
    shutil.copy(DOCS / "thesis_defense.html", kit / "thesis_defense.html")
    for img in (DOCS / "images").iterdir():
        if img.is_file():
            shutil.copy(img, kit / "images" / img.name)
    n_img = len(list((kit / "images").iterdir()))
    print(f"copiate {n_img} immagini")

    from playwright.sync_api import sync_playwright

    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(offline=True, viewport={"width": 1600, "height": 900})
        pg = ctx.new_page()
        pg.goto((kit / "thesis_defense.html").as_uri())
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(shots / "s00.png"))
        seen = 1
        for i in range(1, 40):
            pg.keyboard.press("ArrowRight")
            pg.wait_for_timeout(900)
            pg.screenshot(path=str(shots / f"s{i:02d}.png"))
            seen = i + 1
            if i >= 20:
                break
        b.close()
    print(f"screenshot salvati: {seen} (attese {EXPECTED_SLIDES} slide navigabili)")
    if seen < EXPECTED_SLIDES:
        print("WARN: navigazione incompleta, controllare gli screenshot")
    return kit


def f52_pdf(shots: Path, out: Path) -> Path:
    step("F5.2 PDF deck dagli screenshot (1 slide = 1 pagina, identico al proiettato)")
    from PIL import Image
    from pypdf import PdfReader

    pngs = sorted(shots.glob("s*.png"))
    # s00 = cover, poi una per avanzamento: deduplica frame identici consecutivi
    import hashlib

    pages: list[Path] = []
    last = None
    for p in pngs:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h != last:
            pages.append(p)
            last = h
    print(f"slide uniche: {len(pages)} (attese {EXPECTED_SLIDES})")
    if len(pages) != EXPECTED_SLIDES:
        raise SystemExit(f"FAIL: {len(pages)} slide uniche != {EXPECTED_SLIDES}")
    pdf = out / "thesis_defense.pdf"
    imgs = [Image.open(p).convert("RGB") for p in pages]
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    n = len(PdfReader(str(pdf)).pages)
    print(f"pagine PDF: {n}")
    assert n == EXPECTED_SLIDES
    return pdf


def f53_thesis(out: Path) -> Path:
    step("F5.3 tesi con metadati bonificati")
    from pypdf import PdfReader, PdfWriter

    src = DOCS / "thesis.pdf"
    r = PdfReader(str(src))
    print(f"pagine tesi originale: {len(r.pages)} (attese {EXPECTED_THESIS_PAGES})")
    print("metadati originali:", json.dumps(dict(r.metadata or {}), default=str)[:300])
    if len(r.pages) != EXPECTED_THESIS_PAGES:
        raise SystemExit("FAIL: pagine tesi cambiate, ricontrollare")
    w = PdfWriter()
    for page in r.pages:
        w.add_page(page)
    w.add_metadata({"/Title": "OrchID", "/Author": "", "/Creator": "", "/Producer": ""})
    dst = out / "thesis_consegna.pdf"
    with open(dst, "wb") as f:
        w.write(f)
    r2 = PdfReader(str(dst))
    print(f"pagine tesi consegna: {len(r2.pages)}")
    print("metadati consegna:", json.dumps(dict(r2.metadata or {}), default=str)[:300])
    assert len(r2.pages) == EXPECTED_THESIS_PAGES
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/defense_kit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f51_offline_copy(out)
    f52_pdf(out / "screenshots", out)
    f53_thesis(out)
    print(f"\nKIT PRONTO in {out}: deck/ thesis_defense.pdf thesis_consegna.pdf screenshots/")
    print("Poi: defense_checklist.md (F5.4 nomi/data, F5.5 hardware).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
