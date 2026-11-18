// frontend/src/features/research/sections/DatasetSection.tsx
import { useTranslation } from "react-i18next";
import { Callout } from "@/components/ui/Callout";
import { P, Section, SubSection } from "@/features/research/components/Prose";
import { DataTable } from "@/features/research/components/DataTable";

export function DatasetSection() {
  const { t } = useTranslation();
  return (
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
  );
}
