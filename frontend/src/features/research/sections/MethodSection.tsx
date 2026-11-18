// frontend/src/features/research/sections/MethodSection.tsx
import { useTranslation } from "react-i18next";
import { Figure, P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";

export function MethodSection() {
  const { t } = useTranslation();
  return (
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
          <li><strong>Studio I-A — Cross-Entropy Baseline:</strong>{t("research.metodo_exp1")}</li>
          <li><strong>Studio I-B — Supervised Contrastive Learning:</strong>{t("research.metodo_exp2")}<Cite id="khosla2020supervised" /></li>
          <li><strong>Studio I-C — SpHOR:</strong>{t("research.metodo_exp3")}<Cite id="bahavan2026sphor" /><Cite id="lee2018simple" /><Cite id="liu2020energy" /></li>
          <li><strong>Studio II-A — OSR con Novel Nascosta:</strong>{t("research.metodo_exp4")}</li>
          <li><strong>Studio II-B — OSR con 2 Novel Nascoste:</strong>{t("research.metodo_exp5")}</li>
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
  );
}
