// frontend/src/features/research/sections/Exp2SupconSection.tsx
import { useTranslation } from "react-i18next";
import { Callout } from "@/components/ui/Callout";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";

export function Exp2SupconSection() {
  const { t } = useTranslation();
  return (
    <Section id="exp2" title={t("research.sect_exp2")}>
      <SubSection title={t("research.exp2_subs_motivazione")}>
        <P>
          {t("research.exp2_p1_before")}<Cite id="khosla2020supervised" />{t("research.exp2_p1_after")}<Cite id="liu2024novel" />.
        </P>
      </SubSection>

      <SubSection title={t("research.exp2_subs_protocollo")}>
        <P>
          {t("research.exp2_protocollo")}<Cite id="vaze2022gcd" />.
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
          { model: "ResNet-18", cv: "0,364 ± 0,042", test: "0,287 ± 0,057", gap: "−0,077" },
          { model: "ResNet-50", cv: "0,383 ± 0,037", test: "0,355 ± 0,047", gap: "−0,028" },
          { model: "ConvNeXt-Tiny", cv: "0,844 ± 0,002", test: "0,573 ± 0,027", gap: "−0,271" },
          { model: "ConvNeXt-Small", cv: "0,704 ± 0,198", test: "0,450 ± 0,142", gap: "−0,254" },
          { model: "DINOv2-Small", cv: "0,913 ± 0,009", test: "0,710 ± 0,029", gap: "−0,204" },
          { model: "DINOv2-Base", cv: "0,933 ± 0,006", test: "0,808 ± 0,032", gap: "−0,125" },
        ]}
        highlightRow="DINOv2-Base"
        caption={t("research.exp2_caption")}
      />

      <P>
        {t("research.exp2_risultati")}
      </P>

      <Callout variant="info" title={t("research.exp2_overfitting_title")}>
        {t("research.exp2_overfitting_detail")}
      </Callout>

      <SubSection title={t("research.exp2_subs_embedding_osr")}>
        <P>
          {t("research.exp2_embedding_osr")}
        </P>
      </SubSection>

      <Figure
        src="/figures/per_species_f1.webp"
        alt="F1 per specie"
        caption={t("research.exp2_fig2_caption")}
      />

      <Figure
        src="/figures/metric_curves_dinov2s.webp"
        alt="Metric curves DINOv2-S"
        caption={t("research.exp2_fig3_caption")}
      />
    </Section>
  );
}
