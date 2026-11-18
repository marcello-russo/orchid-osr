// frontend/src/features/research/sections/IntroSection.tsx
import { useTranslation } from "react-i18next";
import { P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";

export function IntroSection() {
  const { t } = useTranslation();
  return (
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
  );
}
