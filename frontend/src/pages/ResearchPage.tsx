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
  { id: "intro", labelKey: "research.sect_intro" },
  { id: "contesto", labelKey: "research.sect_contesto" },
  { id: "dataset", labelKey: "research.sect_dataset" },
  { id: "metodologia", labelKey: "research.sect_metodologia" },
  { id: "exp6", labelKey: "research.sect_exp6" },
  { id: "exp7", labelKey: "research.sect_exp7" },
  { id: "exp8", labelKey: "research.sect_exp8" },
  { id: "exp9", labelKey: "research.sect_exp9" },
  { id: "benchmark", labelKey: "research.sect_benchmark" },
  { id: "conclusioni", labelKey: "research.sect_conclusioni" },
  { id: "bibliografia", labelKey: "research.sect_bibliografia" },
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
              {t(s.labelKey)}
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
        </Section>

        {/* ── 2. DATASET ── */}
        <Section id="dataset" title={t("research.sect_dataset")}>
          <P>
            {t("research.dataset_intro")}
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
              { species: t("research.dataset_row_total"), train: "2.192", test: "300" },
            ]}
            caption={t("research.dataset_caption")}
          />

          <Callout variant="warning" title={t("research.dataset_callout_leakage_title")}>
            {t("research.dataset_callout_leakage")}
          </Callout>

          <Callout variant="info" title={t("research.dataset_callout_novel_title")}>
            {t("research.dataset_callout_novel")}
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
                { model: "ConvNeXt-T", input: "224×224", lr: "10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "ConvNeXt-S", input: "224×224", lr: "10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "DINOv2-S", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
                { model: "DINOv2-B", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
              ]}
              caption={t("research.recipe_caption")}
            />
          </SubSection>

          <SubSection title={t("research.subs_disegno")}>
            <P>
              {t("research.metodo_disegno_intro")}
            </P>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm mb-4">
              <li><strong>Exp 6 — Cross-Entropy Baseline:</strong>{t("research.metodo_exp6")}</li>
              <li><strong>Exp 7 — Supervised Contrastive Learning:</strong>{t("research.metodo_exp7")}<Cite id="khosla2020supervised" /></li>
              <li><strong>Exp 8 — SpHOR:</strong>{t("research.metodo_exp8")}<Cite id="bahavan2026sphor" /><Cite id="lee2018simple" /><Cite id="liu2020energy" /></li>
              <li><strong>Exp 9 — OSR con Novel Nascosta:</strong>{t("research.metodo_exp9")}</li>
            </ul>
          </SubSection>

          <SubSection title={t("research.subs_validazione")}>
            <P>
              {t("research.metodo_validazione")}<Cite id="bendale2015openworld" />
            </P>
          </SubSection>

          <Figure
            src="/figures/architecture.webp"
            alt="Architettura del sistema"
            className="max-h-[720px] w-auto"
            caption={t("research.metodo_arch_fig_caption")}
          />

          <SubSection title={t("research.subs_pipeline_osr")}>
            <P>
              {t("research.metodo_pipeline_osr")}
            </P>
          </SubSection>

          <Figure
            src="/figures/osr-flow.webp"
            alt="Pipeline OSR"
            caption={t("research.metodo_osr_flow_caption")}
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
              { model: "ResNet-18", cv: "0,860", test: "0,978", params: "11M", gap: "+0,117" },
              { model: "ResNet-50", cv: "0,887", test: "0,980", params: "25M", gap: "+0,093" },
              { model: "ConvNeXt-Tiny", cv: "0,702", test: "0,821", params: "28M", gap: "+0,119" },
              { model: "ConvNeXt-Small", cv: "0,527", test: "0,793", params: "50M", gap: "+0,266" },
              { model: "DINOv2-Small", cv: "0,884", test: "0,970", params: "21M", gap: "+0,086" },
              { model: "DINOv2-Base", cv: "0,911", test: "0,975", params: "86M", gap: "+0,064" },
            ]}
            highlightRow="ResNet-50"
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
              { model: "ResNet-18", cv: "0,364", test: "0,444", gap: "+0,080" },
              { model: "ResNet-50", cv: "0,383", test: "0,465", gap: "+0,083" },
              { model: "ConvNeXt-Tiny", cv: "0,844", test: "0,837", gap: "−0,007" },
              { model: "ConvNeXt-Small", cv: "0,704", test: "0,517", gap: "−0,187" },
              { model: "DINOv2-Small", cv: "0,913", test: "0,888", gap: "−0,026" },
              { model: "DINOv2-Base", cv: "0,933", test: "0,936", gap: "+0,004" },
            ]}
            highlightRow="DINOv2-Base"
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
          <SubSection title={t("research.exp8_subs_motivazione")}>
            <P>
              {t("research.exp8_motivazione_before")}<Cite id="bahavan2026sphor" />{t("research.exp8_motivazione_mid")}<Cite id="lee2018simple" />{t("research.exp8_motivazione_after")}<Cite id="liu2020energy" />{t("research.exp8_motivazione_end")}
            </P>
          </SubSection>

          <SubSection title={t("research.exp8_subs_closedset")}>
            <P>
              {t("research.exp8_closedset")}
            </P>
          </SubSection>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "f1t", label: t("research.table_f1_test") },
              { key: "f1cv", label: t("research.table_f1_cv") },
              { key: "knn", label: t("research.table_knn") },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,560", f1cv: "0,473", knn: "0,456" },
              { model: "ResNet-50", f1t: "0,750", f1cv: "0,738", knn: "0,719" },
              { model: "ConvNeXt-Tiny", f1t: "0,911", f1cv: "0,893", knn: "0,884" },
              { model: "ConvNeXt-Small", f1t: "0,874", f1cv: "0,812", knn: "0,879" },
              { model: "DINOv2-Small", f1t: "0,894", f1cv: "0,851", knn: "0,854" },
              { model: "DINOv2-Base", f1t: "0,910", f1cv: "0,896", knn: "0,875" },
            ]}
            highlightRow="ConvNeXt-Tiny"
            caption={t("research.exp8_caption")}
          />

          <SubSection title={t("research.exp8_subs_species")}>
            <P>
              {t("research.exp8_species")}
            </P>
          </SubSection>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/exp8_per_species.webp"
              alt="F1 per specie Exp 8"
              caption={t("research.exp8_per_species_caption")}
            />
            <Figure
              src="/figures/tsne_exp8_multimodel.webp"
              alt="t-SNE multi-modello Exp 8"
              caption={t("research.exp8_tsne_caption")}
            />
          </div>

          <Figure
            src="/figures/confusion_matrices.webp"
            alt="Confusion matrices"
            caption={t("research.exp8_confusion_caption")}
          />
        </Section>

        {/* ── EXP 9: SpHOR con novel class nascosta (O. garganica) ── */}
        <Section id="exp9" title={t("research.sect_exp9")}>
          <SubSection title={t("research.exp9_subs_motivazione")}>
            <P>
              {t("research.exp9_motivazione")}
            </P>
          </SubSection>

          <DataTable
            columns={[
              { key: "model", label: t("research.table_model") },
              { key: "f1t", label: t("research.table_f1_test") },
              { key: "auroc", label: t("research.table_auroc") },
              { key: "det", label: t("research.table_detection") },
              { key: "detector", label: t("research.table_best_detector") },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,590", auroc: "0,507", det: "0,120", detector: "Mahalanobis" },
              { model: "ResNet-50", f1t: "0,856", auroc: "0,776", det: "0,360", detector: "KNN" },
              { model: "ConvNeXt-Tiny", f1t: "0,920", auroc: "0,850", det: "0,640", detector: "Ensemble" },
              { model: "ConvNeXt-Small", f1t: "0,933", auroc: "0,840", det: "0,460", detector: "Ensemble" },
              { model: "DINOv2-Small", f1t: "0,921", auroc: "0,760", det: "0,280", detector: "KNN" },
              { model: "DINOv2-Base", f1t: "0,932", auroc: "0,875", det: "0,560", detector: "Ensemble" },
            ]}
            highlightRow="DINOv2-Base"
            caption={t("research.exp9_caption")}
          />

          <SubSection title={t("research.exp9_subs_osr")}>
            <P>
              {t("research.exp9_osr_intro")}
            </P>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm">
              <li><strong>DINOv2-Base:</strong> {t("research.exp9_osr_dinov2b")}</li>
              <li><strong>ConvNeXt-Tiny:</strong> {t("research.exp9_osr_convnext")}</li>
              <li><strong>ResNet-18:</strong> {t("research.exp9_osr_resnet")}</li>
            </ul>
            <P>
              {t("research.exp9_osr_ensemble")}
            </P>
            <Figure
              src="/figures/exp9_osr_auroc.webp"
              alt="AUROC OSR Exp 9"
              caption={t("research.exp9_osr_auroc_caption")}
            />
          </SubSection>

          <SubSection title={t("research.exp9_subs_clustering")}>
            <P>
              {t("research.exp9_clustering")}
            </P>
            <Figure
              src="/figures/exp9_tsne_novel.webp"
              alt="t-SNE known vs novel Exp 9"
              caption={t("research.exp9_tsne_novel_caption")}
            />
          </SubSection>
        </Section>

        {/* ── 7. BENCHMARK ── */}
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
              { key: "e9", label: t("research.table_e9") },
              { key: "osr9", label: t("research.table_osr9") },
            ]}
            rows={[
              { model: "ResNet-18", e6: "0,978", e7: "0,444", e8: "0,560", e9: "0,590", osr9: "0,507" },
              { model: "ResNet-50", e6: "0,980", e7: "0,465", e8: "0,750", e9: "0,856", osr9: "0,776" },
              { model: "ConvNeXt-Tiny", e6: "0,821", e7: "0,837", e8: "0,911", e9: "0,920", osr9: "0,850" },
              { model: "ConvNeXt-Small", e6: "0,793", e7: "0,517", e8: "0,874", e9: "0,933", osr9: "0,840" },
              { model: "DINOv2-Small", e6: "0,970", e7: "0,888", e8: "0,894", e9: "0,921", osr9: "0,760" },
              { model: "DINOv2-Base", e6: "0,975", e7: "0,936", e8: "0,910", e9: "0,932", osr9: "0,875" },
            ]}
            bests={{ e6: "ResNet-50", e7: "DINOv2-Base", e8: "ConvNeXt-Tiny", e9: "ConvNeXt-Small", osr9: "DINOv2-Base" }}
            caption={t("research.benchmark_caption")}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <Figure
              src="/figures/benchmark_heatmap.webp"
              alt="Benchmark heatmap"
              caption={t("research.benchmark_heatmap_caption")}
            />
            <Figure
              src="/figures/closed_vs_open_scatter.webp"
              alt="Closed vs open"
              caption={t("research.closed_vs_open_caption")}
            />
          </div>

          <SubSection title={t("research.subs_risultati")}>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2">
              <li><strong>{t("research.risultati_li1_before")}</strong>DINOv2-Base{t("research.risultati_li1_mid")}</li>
              <li><strong>{t("research.risultati_li2_before")}</strong>DINOv2-Base{t("research.risultati_li2_mid")}</li>
              <li>{t("research.risultati_li3")}</li>
              <li>{t("research.risultati_li4")}</li>
              <li>{t("research.risultati_li5")}</li>
            </ol>
          </SubSection>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/benchmark_bar.webp"
              alt="Benchmark bar chart"
              caption={t("research.benchmark_bar_caption")}
            />
            <Figure
              src="/figures/summary_dual_heatmap.webp"
              alt="Dual heatmap"
              caption={t("research.summary_dual_heatmap_caption")}
            />
          </div>

          <Figure
            src="/figures/spotlight_species_evolution.webp"
            alt="Species evolution"
            caption={t("research.concl_species_evolution_caption")}
          />

          <Figure
            src="/figures/loss_curves_aggregated.webp"
            alt="Loss curves aggregated"
            caption={t("research.loss_curves_aggregated_caption")}
          />
        </Section>

        {/* ── 9. SINTESI E DISCUSSIONE ── */}
        <Section id="conclusioni" title={t("research.sintesi_title")}>
          <P>
            {t("research.sintesi_implicazioni")}
          </P>
          <P>
            {t("research.sintesi_necessita")}<Cite id="dai2025apl" /><Cite id="wang2025partgcd" />.
          </P>

          <SubSection title={t("research.subs_limiti")}>
            <P>
              {t("research.sintesi_limitazioni")}
            </P>
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
                <span className="text-primary font-mono text-xs shrink-0 mt-0.5">[{p.shortAuthors}, {p.year}]</span>
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
