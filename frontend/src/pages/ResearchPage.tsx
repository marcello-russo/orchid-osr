import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Callout } from "@/components/ui/Callout";
import { Lightbox } from "@/components/ui/Lightbox";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";
import { PAPERS } from "@/features/research/data/papers";
import { cn } from "@/lib/utils";
import { CiteProvider } from "@/features/research/components/CiteContext";

/* ── Sections index ── */

const SECTIONS = [
  { id: "intro", label: "Introduzione" },
  { id: "contesto", label: "Contesto Botanico" },
  { id: "dataset", label: "Dataset" },
  { id: "metodologia", label: "Metodologia" },
  { id: "exp6", label: "Exp 6 — CE" },
  { id: "exp7", label: "Exp 7 — SupCon" },
  { id: "exp8", label: "Exp 8 — SpHOR" },
  { id: "exp10", label: "Exp 10 — MGH-NCD" },
  { id: "benchmark", label: "Benchmark" },
  { id: "conclusioni", label: "Conclusioni" },
  { id: "bibliografia", label: "Bibliografia" },
];

/* ── Hooks ── */

function useActiveSection() {
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

/* ── Shared components ── */

function Figure({ src, alt, caption, className }: { src: string; alt: string; caption?: string; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <figure className="my-8">
      <div className="flex justify-center">
        <img
          src={src} alt={alt}
          className={cn("rounded-xl border border-border cursor-pointer hover:opacity-90 transition-opacity", className || "max-w-full")}
          onClick={() => setOpen(true)}
        />
      </div>
      {caption && <figcaption className="text-sm text-muted-foreground text-center mt-2 italic">{caption}</figcaption>}
      {open && <Lightbox src={src} alt={alt} caption={caption} onClose={() => setOpen(false)} />}
    </figure>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 mb-16">
      <h2 className="text-2xl font-bold tracking-tight mb-6 pb-2 border-b border-border/50">{title}</h2>
      {children}
    </section>
  );
}

function SubSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold tracking-tight mb-3">{title}</h3>
      {children}
    </div>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground leading-relaxed mb-3">{children}</p>;
}

/* ── Main page ── */

export function ResearchPage() {
  const { t } = useTranslation();
  const active = useActiveSection();

  return (
    <CiteProvider>
    <div className="relative max-w-7xl mx-auto px-4 md:px-8">
      {/* Sidebar TOC */}
      <aside className="fixed left-0 top-24 bottom-0 w-64 hidden xl:block overflow-y-auto border-r border-border/50 pr-4 pl-6">
        <nav className="space-y-1 pt-6">
          <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground px-3 pb-2">{t("research.toc_title")}</p>
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={cn(
                "block px-3 py-1.5 text-sm rounded-md transition-colors",
                active === s.id
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
              )}
            >
              {s.label}
            </a>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <main className="research-content relative xl:ml-64 py-12 max-w-4xl">

        {/* ── HERO ── */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary/10 via-primary/5 to-background border border-border/50 mb-12">
          <div className="p-8 md:p-12">
            <Badge variant="outline" className="mb-4 text-xs">{t("research.hero_badge")}</Badge>
            <h1 className="text-3xl md:text-4xl font-black tracking-tight leading-tight">
              {t("research.hero_title")}
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mt-1">
              {t("research.hero_subtitle")}
            </p>
            <div className="mt-6 text-sm text-muted-foreground space-y-1">
              <p><span className="font-medium text-foreground">{t("research.hero_author_label")}</span> Marcello Russo</p>
              <p><span className="font-medium text-foreground">{t("research.hero_advisor_label")}</span> Luigi Libero Lucio Starace</p>
              <p><span className="font-medium text-foreground">{t("research.hero_coadvisor_label")}</span> Mara Sangiovanni, Alessandro De Luca</p>
              <p className="mt-3 italic max-w-2xl text-xs">
                {t("research.hero_university")}
              </p>
            </div>
          </div>
        </div>

        {/* ── INTRODUZIONE ── */}
        <Section id="intro" title={t("research.sect_intro")}>
          <P>
            {t("research.intro_p1_before")}<em>Ophrys</em>{t("research.intro_p1_mid")}<Cite id="yu2021benchmark" />{t("research.intro_p1_after")}
          </P>
          <P>
            {t("research.intro_p2_before")}<Cite id="khosla2020supervised" />{t("research.intro_p2_mid")}<Cite id="bahavan2026sphor" />{t("research.intro_p2_after")}
          </P>
          <SubSection title={t("research.subs_contrib")}>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>{t("research.contrib_1")}</li>
              <li>{t("research.contrib_2")}</li>
              <li>{t("research.contrib_3")}</li>
              <li>{t("research.contrib_4")}<Cite id="liu2020energy" /></li>
            </ul>
          </SubSection>
        </Section>

        {/* ── 1. CONTESTO BOTANICO ── */}
        <Section id="contesto" title={t("research.sect_contesto")}>
          <SubSection title={t("research.subs_genere_ophrys")}>
            <P>
              {t("research.contesto_p1_before")}<em>Ophrys</em>{t("research.contesto_p1_after")}
            </P>
          </SubSection>
          <SubSection title={t("research.subs_mimetismo")}>
            <P>
              {t("research.contesto_p2_before")}<em>Ophrys</em>{t("research.contesto_p2_mid")}<Cite id="schiestl2005swindle" />{t("research.contesto_p2_after")}
            </P>
          </SubSection>
          <SubSection title={t("research.subs_problema_tassonomico")}>
            <P>
              {t("research.contesto_p3_before")}<em>Ophrys</em>{t("research.contesto_p3_mid1")}<Cite id="devey2008ophrys" />{t("research.contesto_p3_mid2")}<em>Flora Europaea</em>{t("research.contesto_p3_mid3")}<em>O. sphegodes</em>{t("research.contesto_p3_mid4")}
            </P>
            <P>
              {t("research.contesto_p4_before")}<em>O. exaltata</em>, <em>O. garganica</em>, <em>O. incubacea</em>, <em>O. majellensis</em>, <em>O. sphegodes</em> e <em>O. sphegodes</em> Palena{t("research.contesto_p4_after")}
            </P>
          </SubSection>
          <Figure
            src="/figures/angular_separability.webp"
            alt="Separabilità angolare"
            caption={t("research.contesto_fig_caption")}
          />
        </Section>

        {/* ── 2. DATASET ── */}
        <Section id="dataset" title={t("research.sect_dataset")}>
          <P>
            Il dataset comprende immagini di sei entità tassonomiche del complesso <em>O. sphegodes</em>,
            tutte fotografate nel territorio italiano, principalmente in Puglia e sul Gargano.
            Le immagini provengono da due fonti: fotografie da campo scattate dai ricercatori
            (fonte primaria per <em>O. majellensis</em> e <em>O. sphegodes</em> Palena) e
            immagini raccolte da banche dati internet aperte per le altre specie.
          </P>

          <DataTable
            columns={[
              { key: "species", label: t("research.dataset_col_species") },
              { key: "train", label: t("research.dataset_col_train") },
              { key: "test", label: t("research.dataset_col_test") },
            ]}
            rows={[
              { species: "O. exaltata", train: "490", test: "50" },
              { species: "O. garganica", train: "563", test: "50" },
              { species: "O. incubacea", train: "352", test: "50" },
              { species: "O. majellensis", train: "152", test: "50" },
              { species: "O. sphegodes", train: "425", test: "50" },
              { species: "O. sphegodes Palena", train: "210", test: "50" },
              { species: "O. sphegodes_Napoli (novel)", train: "0", test: "548" },
              { species: t("research.dataset_row_total"), train: "2.192", test: "848" },
            ]}
            caption={t("research.dataset_caption")}
          />

          <Callout variant="warning" title={t("research.dataset_callout_leakage_title")}>
            {t("research.dataset_callout_leakage")}
          </Callout>

          <Callout variant="info" title={t("research.dataset_callout_novel_title")}>
            Una settima specie, <em>O. sphegodes_Napoli</em>, è stata volutamente esclusa dal training
            set e riservata esclusivamente al test per gli esperimenti di Novel Class Discovery e
            Open-Set Recognition (Exp 8, 10). Morfologicamente affine a <em>O. sphegodes</em> ma con
            caratteri diagnostici distinti a livello di popolazione locale, costituisce il test ideale
            per valutare la capacità del sistema di rilevare entità tassonomiche non precedentemente
            osservate.
          </Callout>

          <SubSection title={t("research.subs_preprocessing")}>
            <P>
              {t("research.dataset_preprocessing")}
            </P>
          </SubSection>
        </Section>

        {/* ── 3. METODOLOGIA ── */}
        <Section id="metodologia" title={t("research.sect_metodologia")}>
          <SubSection title={t("research.subs_architetture")}>
            <P>
              {t("research.arch_before")}<strong>ResNet-18</strong>{t("research.arch_mid1")}<strong>ResNet-50</strong>{t("research.arch_mid2")}<Cite id="he2016deep" />{t("research.arch_mid3")}<strong>ConvNeXt-Tiny</strong>{t("research.arch_mid4")}<strong>ConvNeXt-Small</strong>{t("research.arch_mid5")}<Cite id="liu2022convnet" />{t("research.arch_mid6")}<strong>DINOv2-Small</strong>{t("research.arch_mid7")}<strong>DINOv2-Base</strong>{t("research.arch_mid8")}<Cite id="oquab2024dinov2" />{t("research.arch_after")}
            </P>
          </SubSection>

          <SubSection title={t("research.subs_recipe")}>
            <DataTable
              columns={[
                { key: "model", label: t("research.table_model") },
                { key: "input", label: t("research.table_input") },
                { key: "lr", label: t("research.table_lr") },
                { key: "wd", label: t("research.table_wd") },
                { key: "optim", label: t("research.table_optimizer") },
              ]}
              rows={[
                { model: "ResNet-18", input: "256×512", lr: "10⁻³", wd: "0,0", optim: "Adam" },
                { model: "ResNet-50", input: "256×512", lr: "10⁻³", wd: "0,0", optim: "Adam" },
                { model: "ConvNeXt-T", input: "224×224", lr: "3·10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "ConvNeXt-S", input: "224×224", lr: "3·10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "DINOv2-S", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
                { model: "DINOv2-B", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
              ]}
              caption={t("research.recipe_caption")}
            />
          </SubSection>

          <SubSection title={t("research.subs_disegno")}>
            <P>
              Gli esperimenti sono organizzati in quattro serie, ciascuna delle quali aggiunge
              complessità metodologica alla precedente, in un percorso che va dalla classificazione
              chiusa all&apos;open-world:
            </P>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm mb-4">
              <li><strong>Exp 6 — Cross-Entropy Baseline:</strong> tutte le architetture vengono addestrate con loss CE per 150 epoche con early stopping (patience 15). Stabilisce la baseline di performance closed-set.</li>
              <li><strong>Exp 7 — Supervised Contrastive Learning:</strong> introduce il training in due fasi. Fase 1: backbone + projection head (128-dim) addestrati con SupConLoss (τ=0,07) per 100 epoche, validati con accuratezza 1-NN. Fase 2: backbone congelato, classificatore lineare addestrato con CE per 50 epoche<Cite id="khosla2020supervised" />.</li>
              <li><strong>Exp 8 — SpHOR:</strong> come Exp 7 ma con prototipi apprendibili L2-normalizzati sulla sfera unitaria e loss di repulsione ortogonale (λ=0,01)<Cite id="bahavan2026sphor" />. Al test, quattro rivelatori OSR (Mahalanobis<Cite id="lee2018simple" />, KNN, Energy<Cite id="liu2020energy" />, Ensemble) determinano se un campione è noto o novel.</li>
              <li><strong>Exp 10 — MGH-NCD:</strong> adatta MGCE<Cite id="zheng2025mgce" /> al contesto ultra-FG. Tre esperti di clustering a risoluzioni KNN multiple (k=5, 15, 50) armonizzati tramite divergenza JS.</li>
            </ul>
          </SubSection>

          <SubSection title={t("research.subs_validazione")}>
            <P>
              Tutti gli esperimenti utilizzano cross-validation stratificata a 5 fold con seed 57.
              La metrica principale per la classificazione closed-set è l&apos;F1 macro, robusta allo
              squilibrio naturale del dataset (152 immagini per <em>O. majellensis</em> contro 570
              per <em>O. garganica</em>). Per l&apos;open-set si utilizza il tasso di rilevamento
              al 10% di FPR<Cite id="bendale2015openworld" />: la soglia di rifiuto viene calibrata
              in modo che solo il 10% delle immagini novel venga erroneamente accettato, e si misura
              quanti dei campioni noti vengono correttamente riconosciuti. Un valore di 0,907 significa
              273/300 note riconosciute quando solo 5/50 novel sfuggono.
            </P>
          </SubSection>

          <Figure
            src="/figures/architecture.webp"
            alt="Architettura del sistema"
            className="max-h-[720px] w-auto"
            caption="Diagramma della pipeline di addestramento e valutazione: dalla raccolta immagini alla pubblicazione dei risultati."
          />

          <SubSection title={t("research.subs_pipeline_osr")}>
            <P>
              Per la valutazione OSR, il modello addestrato su 6 classi viene testato su un set
              che include <em>O. sphegodes_Napoli</em> come classe sconosciuta. La procedura si
              compone di tre fasi: (1) estrazione degli embedding dal training e test set tramite
              backbone congelato; (2) addestramento di quattro rivelatori (Mahalanobis, KNN, Energy,
              Ensemble) sui soli embedding noti; (3) calcolo del tasso di rilevamento al 10% FPR.
            </P>
          </SubSection>

          <Figure
            src="/figures/osr-flow.webp"
            alt="Pipeline OSR"
            caption="Pipeline di Open-Set Recognition: estrazione embedding, addestramento rivelatori, rilevamento novità. I rivelatori valutano la distanza/probabilità che un campione appartenga a una classe nota."
          />
        </Section>

        {/* ── 4. EXP 6: CE ── */}
        <Section id="exp6" title={t("research.sect_exp6")}>
          <SubSection title={t("research.exp6_subs_motivazione")}>
            <P>
              {t("research.exp6_p1_before")}<Cite id="geng2020openset" />.
            </P>
            <P>
              {t("research.exp6_training")}
            </P>
          </SubSection>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "cv", label: t("research.table_f1_cv") },
              { key: "test", label: t("research.table_f1_test") },
              { key: "params", label: t("research.table_params") },
              { key: "gap", label: t("research.table_gap") },
            ]}
            rows={[
              { model: "ResNet-18", cv: "0,860", test: "0,880", params: "11M", gap: "+0,020" },
              { model: "ResNet-50", cv: "0,858", test: "0,882", params: "25M", gap: "+0,024" },
              { model: "ConvNeXt-Tiny", cv: "0,892", test: "0,890", params: "28M", gap: "−0,002" },
              { model: "ConvNeXt-Small", cv: "0,883", test: "0,863", params: "50M", gap: "−0,020" },
              { model: "DINOv2-Small", cv: "0,906", test: "0,888", params: "21M", gap: "−0,018" },
              { model: "DINOv2-Base", cv: "0,916", test: "0,927", params: "86M", gap: "+0,011" },
            ]}
            highlightRow="DINOv2-Base"
            caption={t("research.exp6_caption")}
          />

          <P>
            {t("research.exp6_risultati_before")}<em>double descent</em>{t("research.exp6_risultati_mid")}
          </P>
          <P>
            {t("research.exp6_risultati_after")}
          </P>

          <Figure
            src="/figures/cv_test_gap.webp"
            alt="Generalization gap"
            caption={t("research.exp6_fig_cv_gap")}
          />

          <Figure
            src="/figures/loss_curves_exp6_dinov2b.webp"
            alt="Loss curves DINOv2-B Exp 6"
            caption={t("research.exp6_fig_caption")}
          />

          <SubSection title={t("research.exp6_subs_embedding")}>
            <P>
              {t("research.exp6_embedding")}<Cite id="vaze2022gcd" />.
            </P>
          </SubSection>
        </Section>

        {/* ── 5. EXP 7: SUPCON ── */}
        <Section id="exp7" title={t("research.sect_exp7")}>
          <SubSection title={t("research.exp7_subs_motivazione")}>
            <P>
              {t("research.exp7_p1_before")}<Cite id="khosla2020supervised" />{t("research.exp7_p1_after")}<Cite id="liu2024novel" />.
            </P>
          </SubSection>

          <SubSection title={t("research.exp7_subs_protocollo")}>
            <P>
              {t("research.exp7_protocollo")}<Cite id="vaze2022gcd" />.
            </P>
          </SubSection>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "cv", label: t("research.table_f1_cv") },
              { key: "test", label: t("research.table_f1_test") },
              { key: "gap", label: t("research.table_gap") },
            ]}
            rows={[
              { model: "ResNet-18", cv: "0,612", test: "0,736", gap: "+0,123" },
              { model: "ResNet-50", cv: "0,736", test: "0,799", gap: "+0,063" },
              { model: "ConvNeXt-Tiny", cv: "0,876", test: "0,681", gap: "−0,195" },
              { model: "ConvNeXt-Small", cv: "0,882", test: "0,689", gap: "−0,194" },
              { model: "DINOv2-Small", cv: "0,906", test: "0,896", gap: "−0,010" },
              { model: "DINOv2-Base", cv: "0,914", test: "0,687", gap: "−0,226" },
            ]}
            highlightRow="DINOv2-Small"
            caption={t("research.exp7_caption")}
          />

          <P>
            {t("research.exp7_risultati")}
          </P>

          <Callout variant="info" title={t("research.exp7_overfitting_title")}>
            {t("research.exp7_overfitting_detail")}
          </Callout>

          <SubSection title={t("research.exp7_subs_embedding_osr")}>
            <P>
              {t("research.exp7_embedding_osr")}
            </P>
          </SubSection>

          <Figure
            src="/figures/cv_test_gap.webp"
            alt="Generalization gap"
            caption={t("research.exp7_fig1_caption")}
          />

          <Figure
            src="/figures/per_species_f1.webp"
            alt="F1 per specie"
            caption={t("research.exp7_fig2_caption")}
          />

          <Figure
            src="/figures/metric_curves_dinov2s.webp"
            alt="Metric curves DINOv2-S"
            caption={t("research.exp7_fig3_caption")}
          />
        </Section>

        {/* ── 6. EXP 8: SPHOR ── */}
        <Section id="exp8" title={t("research.sect_exp8")}>
          <P>
            {t("research.exp8_p1_before")}<Cite id="bahavan2026sphor" />{t("research.exp8_p1_mid")}<Cite id="lee2018simple" />{t("research.exp8_p1_mid2")}<Cite id="liu2020energy" />{t("research.exp8_p1_after")}
          </P>

          <Callout variant="info" title={t("research.exp8_callout_title")}>
            {t("research.exp8_callout_body")}
          </Callout>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "f1t", label: t("research.table_f1_test") },
              { key: "f1cv", label: t("research.table_f1_cv") },
              { key: "det", label: t("research.table_detection") },
              { key: "detector", label: t("research.table_best_detector") },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,520", f1cv: "0,398", det: "0,093", detector: "Mahalanobis" },
              { model: "ResNet-50", f1t: "0,609", f1cv: "0,736", det: "0,538", detector: "Ensemble" },
              { model: "ConvNeXt-Tiny", f1t: "0,759", f1cv: "0,794", det: "0,907", detector: "Ensemble" },
              { model: "ConvNeXt-Small", f1t: "0,668", f1cv: "0,735", det: "0,480", detector: "Energy" },
              { model: "DINOv2-Small", f1t: "0,747", f1cv: "0,906", det: "0,670", detector: "Ensemble" },
              { model: "DINOv2-Base", f1t: "0,654", f1cv: "0,870", det: "0,746", detector: "Energy" },
            ]}
            highlightRow="ConvNeXt-Tiny"
            caption={t("research.exp8_caption")}
          />

          <Callout variant="success" title={t("research.exp8_winner_callout_title")}>
            {t("research.exp8_winner_body")}
          </Callout>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/spotlight_convnext_osr.webp"
              alt="Detection rate per rivelatore"
              caption="Tasso di rilevamento per rivelatore. ConvNeXt-Tiny (arancione) raggiunge 0,907 con Ensemble."
            />
            <Figure
              src="/figures/exp8_per_species.webp"
              alt="F1 per specie Exp 8"
              caption="F1 per specie in Exp 8: O. sphegodes è la più penalizzata; Palena e Exaltata le meno colpite."
            />
          </div>

          <P>
            L&apos;analisi del clustering sugli embedding SpHOR conferma che la separazione
            non supervisionata rimane aperta: l&apos;NMI al k reale (k=6) è 0,164 per ConvNeXt-Tiny
            e 0,187 per DINOv2-Base. K-means converge a k=2 (separazione binaria tra i due cluster
            principali). DBSCAN classifica il 96% dei punti come rumore.
          </P>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/osr_detection_rate.webp"
              alt="OSR detection rate Exp 8"
              caption="Detection rate al variare della soglia FPR per ogni rivelatore in Exp 8. Ensemble e KNN sono i più robusti."
            />
            <Figure
              src="/figures/clustering_comparison.webp"
              alt="Clustering comparison"
              caption="Confronto del clustering (K-means, Spectral, DBSCAN) su embedding SpHOR vs MGH-NCD. Nessun metodo recupera le sei specie."
            />
          </div>

          <Figure
            src="/figures/tsne_exp8_multimodel.webp"
            alt="t-SNE multi-modello Exp 8"
            caption="t-SNE per tre architetture in Exp 8: (a) ConvNeXt-Tiny — embedding ben separati, outlier presenti; (b) DINOv2-Small — cluster più coesi ma meno separati; (c) DINOv2-Base — struttura intermedia."
          />
        </Section>

        {/* ── 7. EXP 10: MGH-NCD ── */}
        <Section id="exp10" title={t("research.sect_exp10")}>
          <P>
            {t("research.exp10_p1_before")}<Cite id="zheng2025mgce" />{t("research.exp10_p1_after")}
          </P>

          <Callout variant="danger" title={t("research.exp10_callout_title")}>
            {t("research.exp10_callout_body")}<em>O. sphegodes</em>{t("research.exp10_callout_body_mid")}
          </Callout>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "f1t", label: t("research.table_f1_test") },
              { key: "det", label: t("research.table_detection") },
              { key: "nmi", label: t("research.table_nmi") },
              { key: "acc", label: t("research.table_acc_clust") },
              { key: "palena", label: t("research.table_palena") },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,226", det: "0,257", nmi: "0,034", acc: "0,253", palena: "64%" },
              { model: "ResNet-50", f1t: "0,174", det: "0,321", nmi: "0,023", acc: "0,206", palena: "68%" },
              { model: "ConvNeXt-Tiny", f1t: "0,349", det: "0,566", nmi: "0,142", acc: "0,386", palena: "55%" },
              { model: "ConvNeXt-Small", f1t: "0,247", det: "0,536", nmi: "0,075", acc: "0,330", palena: "60%" },
              { model: "DINOv2-Small", f1t: "0,270", det: "0,391", nmi: "0,239", acc: "0,592", palena: "61%" },
              { model: "DINOv2-Base", f1t: "0,369", det: "0,688", nmi: "0,485", acc: "0,511", palena: "74%" },
            ]}
            highlightRow="DINOv2-Base"
            caption={t("research.exp10_caption")}
          />

          <Figure
            src="/figures/spotlight_exp10_collapse.webp"
            alt="Collasso Exp 10"
            caption="Collasso di Exp 10 per DINOv2-Base. (a) Matrice di confusione: la colonna Palena cattura la maggior parte delle predizioni. (b) Per ogni specie, la frazione Palena supera l'accuratezza corretta."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/confusion_matrices.webp"
              alt="Confusion matrices"
              caption="Matrici di confusione per Exp 7 (SupCon), Exp 8 (SpHOR) e Exp 10 (MGH-NCD). Il degrado progressivo è evidente: da una diagonale pulita (SupCon) al collasso su Palena (MGH-NCD)."
            />
            <Figure
              src="/figures/tsne_novel_exp10.webp"
              alt="t-SNE with novel Exp 10"
              caption="Proiezione t-SNE degli embedding di Exp 10 con la classe novel O. sphegodes_Napoli (cerchi rossi). La novel si sovrappone completamente alle classi note."
            />
          </div>

          <Figure
            src="/figures/exp10_per_species.webp"
            alt="F1 per specie Exp 10"
            caption="F1 per specie in Exp 10. Nessuna specie raggiunge 0,5 F1. Il collasso è uniforme: tutte le specie soffrono allo stesso modo."
          />

          <SubSection title={t("research.subs_analisi")}>
            <P>
              Il fallimento si articola su tre livelli. (1) <strong>Consensus spuria</strong>:
              l&apos;allineamento degli esperti forza il consenso tra tre risoluzioni KNN; in
              ultra-FGVC, dove la risoluzione &ldquo;grossolana&rdquo; (k=50) cattura già solo
              differenze finissime, il consenso collassa sulla classe più distintiva (Palena).
              (2) <strong>Full FT impatta negativamente</strong>: a differenza di Exp 8, dove
              LoRA regolarizzava, in Exp 10 la maggiore capacità ha amplificato il collasso.
              (3) <strong>Complessità eccessiva</strong>: tre esperti, memory bank, tre loss per
              2.232 immagini — il modello &ldquo;spende&rdquo; la sua capacità per soddisfare i
              vincoli di allineamento.
            </P>
          </SubSection>

          <Callout variant="info" title="Lezione: la semplicità supera la complessità">
            SpHOR, con un singolo obiettivo contrastivo e prototipi ortogonali, supera MGCE
            nonostante la minore sofisticazione. La scelta dell&apos;approccio deve essere guidata
            dalla scala intrinseca del problema, non dalla sofisticazione della tecnica. Questo è
            un risultato scientifico valido: dimostra che l&apos;aggiunta di complessità non porta
            necessariamente a miglioramenti in ultra-FGVC.
          </Callout>
        </Section>

        {/* ── 8. BENCHMARK ── */}
        <Section id="benchmark" title={t("research.sect_benchmark")}>
          <P>
            {t("research.benchmark_intro")}
          </P>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "e6", label: t("research.table_e6") },
              { key: "e7", label: t("research.table_e7") },
              { key: "e8", label: t("research.table_e8") },
              { key: "e10", label: t("research.table_e10") },
              { key: "osr8", label: t("research.table_osr8") },
              { key: "osr10", label: t("research.table_osr10") },
            ]}
            rows={[
              { model: "ResNet-18", e6: "0,880", e7: "0,736", e8: "0,520", e10: "0,226", osr8: "0,093", osr10: "0,257" },
              { model: "ResNet-50", e6: "0,882", e7: "0,799", e8: "0,609", e10: "0,174", osr8: "0,538", osr10: "0,321" },
              { model: "ConvNeXt-Tiny", e6: "0,890", e7: "0,681", e8: "0,759", e10: "0,349", osr8: "0,907", osr10: "0,566" },
              { model: "ConvNeXt-Small", e6: "0,863", e7: "0,689", e8: "0,668", e10: "0,247", osr8: "0,480", osr10: "0,536" },
              { model: "DINOv2-Small", e6: "0,888", e7: "0,896", e8: "0,747", e10: "0,270", osr8: "0,670", osr10: "0,391" },
              { model: "DINOv2-Base", e6: "0,927", e7: "0,687", e8: "0,654", e10: "0,369", osr8: "0,746", osr10: "0,688" },
            ]}
            bests={{ e6: "DINOv2-Base", e7: "DINOv2-Small", osr8: "ConvNeXt-Tiny" }}
            caption={t("research.benchmark_caption")}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <Figure
              src="/figures/benchmark_heatmap.webp"
              alt="Benchmark heatmap"
              caption="Heatmap: F1 test per modello ed esperimento."
            />
            <Figure
              src="/figures/closed_vs_open_scatter.webp"
              alt="Closed vs open"
              caption="F1 test vs Detection Rate: il trade-off closed-set / open-set."
            />
          </div>

          <SubSection title={t("research.subs_risultati")}>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2">
              <li><strong>{t("research.risultati_li1_before")}</strong>DINOv2-Small{t("research.risultati_li1_mid")}</li>
              <li><strong>{t("research.risultati_li2_before")}</strong>ConvNeXt-Tiny{t("research.risultati_li2_mid")}</li>
              <li>{t("research.risultati_li3")}</li>
              <li>{t("research.risultati_li4")}</li>
              <li>{t("research.risultati_li5")}</li>
            </ol>
          </SubSection>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/benchmark_bar.webp"
              alt="Benchmark bar chart"
              caption="Confronto a barre dell'F1 test attraverso esperimenti e architetture."
            />
            <Figure
              src="/figures/summary_dual_heatmap.webp"
              alt="Dual heatmap"
              caption="Heatmap duale: F1 test (sinistra) e Detection Rate (destra) affiancati per confronto immediato."
            />
          </div>

          <Figure
            src="/figures/osr_detection_comparison.webp"
            alt="OSR detection comparison"
            caption="Detection rate al 10% FPR per tutti i rivelatori: Exp 8 (sinistra) vs Exp 10 (destra)."
          />

          <Figure
            src="/figures/spotlight_species_evolution.webp"
            alt="Species evolution"
            caption="Evoluzione dell'F1 per specie attraverso i quattro esperimenti. Il degrado è generalizzato, ma alcune specie (O. sphegodes, O. garganica) soffrono più di altre."
          />

          <Figure
            src="/figures/loss_curves_aggregated.webp"
            alt="Loss curves aggregated"
            caption="Curve di loss aggregate per esperimento. La convergenza è più rapida in Exp 6 (CE) rispetto a Exp 7–10 (contrastive)."
          />
        </Section>

        {/* ── 9. CONCLUSIONI ── */}
        <Section id="conclusioni" title={t("research.sect_conclusioni")}>
          <P>
            {t("research.conclusioni_p1_before")}<em>Ophrys</em>{t("research.conclusioni_p1_after")}
          </P>

          <SubSection title={t("research.subs_conclusioni_contrib")}>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>{t("research.contrib_li1")}<em>Ophrys</em>{t("research.contrib_li1_end")}</li>
              <li>{t("research.contrib_li2")}</li>
              <li>{t("research.contrib_li3")}</li>
              <li>{t("research.contrib_li4")}<em>Ophrys</em>{t("research.contrib_li4_end")}</li>
            </ul>
          </SubSection>

          <SubSection title={t("research.subs_limiti")}>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>{t("research.limiti_li1")}</li>
              <li>{t("research.limiti_li2")}</li>
              <li>{t("research.limiti_li3")}</li>
              <li>{t("research.limiti_li4")}</li>
            </ul>
          </SubSection>

          <SubSection title={t("research.subs_futuri")}>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>{t("research.futuri_li1")}</li>
              <li>{t("research.futuri_li2_before")}<Cite id="dai2025apl" />{t("research.futuri_li2_after")}<Cite id="wang2025partgcd" />{t("research.futuri_li2_end")}</li>
              <li>{t("research.futuri_li3")}</li>
              <li>{t("research.futuri_li4")}</li>
            </ul>
          </SubSection>
        </Section>

        {/* ── 10. BIBLIOGRAFIA ── */}
        <Section id="bibliografia" title={t("research.sect_bibliografia")}>
          <div className="space-y-3 text-sm">
            {PAPERS.map((p) => (
              <div key={p.id} className="flex gap-2">
                <span className="text-primary font-mono text-xs shrink-0 mt-0.5">[{p.id}]</span>
                <div>
                  <span className="font-medium text-foreground">{p.authors} ({p.year}).</span>{" "}
                  <em>{p.title}.</em>{" "}
                  {p.venue && <span>{p.venue}. </span>}
                  {p.url && (
                    <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                      [Link]
                    </a>
                  )}
                  {p.doi && (
                    <a href={`https://doi.org/${p.doi}`} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                      [DOI]
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Section>

      </main>
    </div>
    </CiteProvider>
  );
}
