// frontend/src/features/research/sections/ConclusionsSection.tsx
import { useTranslation } from "react-i18next";
import { P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";

export function ConclusionsSection() {
  const { t } = useTranslation();
  return (
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
  );
}
