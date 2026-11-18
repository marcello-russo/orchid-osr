// frontend/src/features/research/sections/BenchmarkSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { DataTable } from "@/features/research/components/DataTable";

export function BenchmarkSection() {
  const { t } = useTranslation();
  return (
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
          { key: "nmi4", label: "NMI" },
        ]}
        rows={[
          { model: "ResNet-18", e6: "0,847", e7: "0,287", e8: "0,442", e9: "0,525", osr9: "0,507", nmi4: "0,752" },
          { model: "ResNet-50", e6: "0,884", e7: "0,355", e8: "0,719", e9: "0,784", osr9: "0,776", nmi4: "0,846" },
          { model: "ConvNeXt-Tiny", e6: "0,687", e7: "0,573", e8: "0,710", e9: "0,929", osr9: "0,850", nmi4: "0,921" },
          { model: "ConvNeXt-Small", e6: "0,513", e7: "0,450", e8: "0,677", e9: "0,930", osr9: "0,839", nmi4: "0,910" },
          { model: "DINOv2-Small", e6: "0,695", e7: "0,710", e8: "0,772", e9: "0,924", osr9: "0,758", nmi4: "0,922" },
          { model: "DINOv2-Base", e6: "0,764", e7: "0,808", e8: "0,833", e9: "0,929", osr9: "0,875", nmi4: "0,928" },
        ]}
        bests={{ e6: "ResNet-50", e7: "DINOv2-Base", e8: "DINOv2-Base", e9: "ConvNeXt-Small", osr9: "DINOv2-Base", nmi4: "DINOv2-Base" }}
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
  );
}
