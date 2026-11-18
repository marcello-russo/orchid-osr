// frontend/src/features/research/sections/Exp3SphorSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";

export function Exp3SphorSection() {
  const { t } = useTranslation();
  return (
    <Section id="exp3" title={t("research.sect_exp3")}>
      <SubSection title={t("research.exp3_subs_motivazione")}>
        <P>
          {t("research.exp3_motivazione_before")}<Cite id="bahavan2026sphor" />{t("research.exp3_motivazione_mid")}<Cite id="lee2018simple" />{t("research.exp3_motivazione_after")}<Cite id="liu2020energy" />{t("research.exp3_motivazione_end")}
        </P>
      </SubSection>

      <SubSection title={t("research.exp3_subs_closedset")}>
        <P>
          {t("research.exp3_closedset")}
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
          { model: "ResNet-18", f1t: "0,442 ± 0,060", f1cv: "0,473 ± 0,065", knn: "0,456" },
          { model: "ResNet-50", f1t: "0,719 ± 0,041", f1cv: "0,738 ± 0,035", knn: "0,719" },
          { model: "ConvNeXt-Tiny", f1t: "0,710 ± 0,040", f1cv: "0,893 ± 0,009", knn: "0,884" },
          { model: "ConvNeXt-Small", f1t: "0,677 ± 0,046", f1cv: "0,812 ± 0,102", knn: "0,879" },
          { model: "DINOv2-Small", f1t: "0,772 ± 0,039", f1cv: "0,851 ± 0,020", knn: "0,854" },
          { model: "DINOv2-Base", f1t: "0,833 ± 0,031", f1cv: "0,896 ± 0,013", knn: "0,875" },
        ]}
        highlightRow="DINOv2-Base"
        caption={t("research.exp3_caption")}
      />

      <SubSection title={t("research.exp3_subs_species")}>
        <P>
          {t("research.exp3_species")}
        </P>
      </SubSection>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Figure
          src="/figures/exp8_per_species.webp"
          alt="F1 per specie Studio I-C"
          caption={t("research.exp3_per_species_caption")}
        />
        <Figure
          src="/figures/tsne_exp8_multimodel.webp"
          alt="t-SNE multi-modello Studio I-C"
          caption={t("research.exp3_tsne_caption")}
        />
      </div>

      <Figure
        src="/figures/confusion_matrices.webp"
        alt="Confusion matrices"
        caption={t("research.exp3_confusion_caption")}
      />
    </Section>
  );
}
