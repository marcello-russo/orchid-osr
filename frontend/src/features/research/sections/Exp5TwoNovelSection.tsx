// frontend/src/features/research/sections/Exp5TwoNovelSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { DataTable } from "@/features/research/components/DataTable";

export function Exp5TwoNovelSection() {
  const { t } = useTranslation();
  return (
    <Section id="exp5" title={t("research.sect_exp5")}>
      <SubSection title={t("research.exp5_subs_motivazione")}>
        <P>
          {t("research.exp5_motivazione")}
        </P>
      </SubSection>

      <DataTable
        columns={[
          { key: "model", label: t("research.table_model") },
          { key: "cv", label: t("research.table_f1_cv") },
          { key: "test", label: t("research.table_f1_test") },
        ]}
        rows={[
          { model: "ResNet-18", cv: "0,813 ± 0,020", test: "0,810 ± 0,012" },
          { model: "ResNet-50", cv: "0,816 ± 0,013", test: "0,878 ± 0,018" },
          { model: "ConvNeXt-Tiny", cv: "0,939 ± 0,015", test: "0,923 ± 0,009" },
          { model: "ConvNeXt-Small", cv: "0,932 ± 0,009", test: "0,916 ± 0,004" },
          { model: "DINOv2-Small", cv: "0,913 ± 0,025", test: "0,923 ± 0,007" },
          { model: "DINOv2-Base", cv: "0,928 ± 0,016", test: "0,930 ± 0,004" },
        ]}
        highlightRow="DINOv2-Base"
        caption={t("research.exp5_caption")}
      />

      <SubSection title={t("research.exp5_subs_osr")}>
        <P>
          {t("research.exp5_osr")}
        </P>
        <Figure
          src="/figures/osr_1vs2_novel.webp"
          alt="OSR 1 vs 2 novel"
          caption={t("research.osr_1vs2_caption")}
        />
      </SubSection>

      <DataTable
        columns={[
          { key: "model", label: t("research.table_model") },
          { key: "maha", label: "Mahalanobis" },
          { key: "knn", label: "KNN" },
          { key: "en", label: "Energy" },
          { key: "ens", label: "Ensemble*" },
        ]}
        rows={[
          { model: "ResNet-18", maha: "0,604", knn: "0,664", en: "0,559", ens: "0,666" },
          { model: "ResNet-50", maha: "0,621", knn: "0,689", en: "0,440", ens: "0,634" },
          { model: "ConvNeXt-Tiny", maha: "0,849", knn: "0,836", en: "0,703", ens: "0,843" },
          { model: "ConvNeXt-Small", maha: "0,785", knn: "0,774", en: "0,410", ens: "0,718" },
          { model: "DINOv2-Small", maha: "0,723", knn: "0,787", en: "0,505", ens: "0,734" },
          { model: "DINOv2-Base", maha: "0,835", knn: "0,835", en: "0,488", ens: "0,805" },
        ]}
        highlightRow="ConvNeXt-Tiny"
        caption={t("research.exp5_osr_caption")}
      />

      <SubSection title={t("research.exp5_subs_gcd")}>
        <P>
          {t("research.exp5_gcd")}
        </P>
        <Figure
          src="/figures/gcd_old_new.webp"
          alt="GCD 1 vs 2 novel"
          caption={t("research.gcd_exp5_caption")}
        />
        <P>
          {t("research.exp5_delong")}
        </P>
      </SubSection>
    </Section>
  );
}
