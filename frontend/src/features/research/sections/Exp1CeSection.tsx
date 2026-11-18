// frontend/src/features/research/sections/Exp1CeSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";

export function Exp1CeSection() {
  const { t } = useTranslation();
  return (
    <Section id="exp1" title={t("research.sect_exp1")}>
      <SubSection title={t("research.exp1_subs_motivazione")}>
        <P>
          {t("research.exp1_p1_before")}<Cite id="geng2020openset" />.
        </P>
        <P>
          {t("research.exp1_training")}
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
          { model: "ResNet-18", cv: "0,860 ± 0,016", test: "0,847 ± 0,016", params: "11M", gap: "−0,013" },
          { model: "ResNet-50", cv: "0,887 ± 0,009", test: "0,884 ± 0,009", params: "25M", gap: "−0,003" },
          { model: "ConvNeXt-Tiny", cv: "0,702 ± 0,026", test: "0,687 ± 0,032", params: "28M", gap: "−0,016" },
          { model: "ConvNeXt-Small", cv: "0,527 ± 0,171", test: "0,513 ± 0,164", params: "50M", gap: "−0,014" },
          { model: "DINOv2-Small", cv: "0,884 ± 0,009", test: "0,695 ± 0,041", params: "21M", gap: "−0,189" },
          { model: "DINOv2-Base", cv: "0,911 ± 0,003", test: "0,764 ± 0,058", params: "86M", gap: "−0,147" },
        ]}
        highlightRow="ResNet-50"
        caption={t("research.exp1_caption")}
      />

      <P>
        {t("research.exp1_risultati_before")}<em>double descent</em>{t("research.exp1_risultati_mid")}
      </P>
      <P>
        {t("research.exp1_risultati_after")}
      </P>

      <Figure
        src="/figures/cv_test_gap.webp"
        alt="Generalization gap"
        caption={t("research.exp1_fig_cv_gap")}
      />

      <Figure
        src="/figures/loss_curves_exp6_dinov2b_v2.webp"
        alt="Loss curves DINOv2-B Studio I-A"
        caption={t("research.exp1_fig_caption")}
      />

      <SubSection title={t("research.exp1_subs_embedding")}>
        <P>
          {t("research.exp1_embedding")}<Cite id="vaze2022gcd" />.
        </P>
      </SubSection>
    </Section>
  );
}
