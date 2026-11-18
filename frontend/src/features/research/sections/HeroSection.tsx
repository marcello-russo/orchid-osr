// frontend/src/features/research/sections/HeroSection.tsx
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";

export function HeroSection() {
  const { t } = useTranslation();
  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary/10 via-primary/5 to-background border border-border/50 mb-12">
      <div className="p-8 md:p-12">
        <Badge variant="outline" className="mb-4 text-xs">{t("research.hero_badge")}</Badge>
        <h1 className="text-3xl md:text-4xl font-black tracking-tight leading-tight">
          {t("research.hero_title")}
        </h1>
        <p className="text-lg md:text-xl text-muted-foreground mt-1">
          {t("research.hero_subtitle")}
        </p>
        <div className="mt-6 text-sm text-muted-foreground space-y-1">
          <p><span className="font-medium text-foreground">{t("research.hero_author_label")}</span> Marcello Russo</p>
          <p><span className="font-medium text-foreground">{t("research.hero_advisor_label")}</span> Prof. Sergio Di Martino, Prof.ssa Mara Sangiovanni, Prof. Luigi Libero Lucio Starace</p>
          <p><span className="font-medium text-foreground">{t("research.hero_coadvisor_label")}</span> Prof. Alessandro De Luca</p>
          <p className="mt-3 italic max-w-2xl text-xs">
            {t("research.hero_university")}
          </p>
        </div>
      </div>
    </div>
  );
}
