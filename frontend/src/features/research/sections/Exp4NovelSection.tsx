// frontend/src/features/research/sections/Exp4NovelSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { DataTable } from "@/features/research/components/DataTable";

export function Exp4NovelSection() {
  const { t } = useTranslation();
  return (
    <Section id="exp4" title={t("research.sect_exp4")}>
      <SubSection title={t("research.exp4_subs_motivazione")}>
        <P>
          {t("research.exp4_motivazione")}
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
          { model: "ResNet-18", f1t: "0,525 ± 0,072", auroc: "0,507", det: "0,120", detector: "Mahalanobis" },
          { model: "ResNet-50", f1t: "0,784 ± 0,049", auroc: "0,776", det: "0,360", detector: "KNN" },
          { model: "ConvNeXt-Tiny", f1t: "0,929 ± 0,007", auroc: "0,850", det: "0,640", detector: "Ensemble" },
          { model: "ConvNeXt-Small", f1t: "0,930 ± 0,007", auroc: "0,839", det: "0,460", detector: "Ensemble" },
          { model: "DINOv2-Small", f1t: "0,924 ± 0,012", auroc: "0,758", det: "0,280", detector: "KNN" },
          { model: "DINOv2-Base", f1t: "0,929 ± 0,005", auroc: "0,875", det: "0,540", detector: "Ensemble" },
        ]}
        highlightRow="DINOv2-Base"
        caption={t("research.exp4_caption")}
      />

      <SubSection title={t("research.exp4_subs_osr")}>
        <P>
          {t("research.exp4_osr_intro")}
        </P>
        <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm">
          <li><strong>DINOv2-Base:</strong> {t("research.exp4_osr_dinov2b")}</li>
          <li><strong>ConvNeXt-Tiny:</strong> {t("research.exp4_osr_convnext")}</li>
          <li><strong>ResNet-18:</strong> {t("research.exp4_osr_resnet")}</li>
        </ul>
        <P>
          {t("research.exp4_osr_ensemble")}
        </P>
        <Figure
          src="/figures/exp9_osr_auroc.webp"
          alt="AUROC OSR Studio II-A"
          caption={t("research.exp4_osr_auroc_caption")}
        />
      </SubSection>

      <SubSection title={t("research.exp4_subs_clustering")}>
        <P>
          {t("research.exp4_clustering")}
        </P>
        <Figure
          src="/figures/exp9_tsne_novel.webp"
          alt="t-SNE known vs novel Studio II-A"
          caption={t("research.exp4_tsne_novel_caption")}
        />
      </SubSection>
    </Section>
  );
}
