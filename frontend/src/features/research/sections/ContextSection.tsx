// frontend/src/features/research/sections/ContextSection.tsx
import { useTranslation } from "react-i18next";
import { P, Section, SubSection } from "@/features/research/components/Prose";
import { Cite } from "@/features/research/components/Citation";

export function ContextSection() {
  const { t } = useTranslation();
  return (
    <Section id="contesto" title={t("research.sect_contesto")}>
      <SubSection title={t("research.subs_genere_ophrys")}>
        <P>
          {t("research.contesto_p1_before")}<em>Ophrys</em>{t("research.contesto_p1_after")}
        </P>
      </SubSection>
      <SubSection title={t("research.subs_mimetismo")}>
        <P>
          {t("research.contesto_p2_before")}<em>Ophrys</em>{t("research.contesto_p2_mid")}<Cite id="schiestl2005swindle" />{t("research.contesto_p2_after")}
        </P>
      </SubSection>
      <SubSection title={t("research.subs_problema_tassonomico")}>
        <P>
          {t("research.contesto_p3_before")}<em>Ophrys</em>{t("research.contesto_p3_mid1")}<Cite id="devey2008ophrys" />{t("research.contesto_p3_mid2")}<em>Flora Europaea</em>{t("research.contesto_p3_mid3")}<em>O. sphegodes</em>{t("research.contesto_p3_mid4")}
        </P>
        <P>
          {t("research.contesto_p4_before")}<em>O. exaltata</em>, <em>O. garganica</em>, <em>O. incubacea</em>, <em>O. majellensis</em>, <em>O. sphegodes</em> e <em>O. sphegodes</em> Palena{t("research.contesto_p4_after")}
        </P>
      </SubSection>
    </Section>
  );
}
