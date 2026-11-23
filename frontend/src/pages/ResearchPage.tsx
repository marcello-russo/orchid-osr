import { useTranslation } from "react-i18next";
import { CiteProvider } from "@/features/research/components/CiteContext";
import { useActiveSection, SECTIONS } from "@/features/research/hooks/useActiveSection";
import { BibliographySection } from "@/features/research/sections/BibliographySection";
import { BenchmarkSection } from "@/features/research/sections/BenchmarkSection";
import { ConclusionsSection } from "@/features/research/sections/ConclusionsSection";
import { ContextSection } from "@/features/research/sections/ContextSection";
import { DatasetSection } from "@/features/research/sections/DatasetSection";
import { Exp1CeSection } from "@/features/research/sections/Exp1CeSection";
import { Exp2SupconSection } from "@/features/research/sections/Exp2SupconSection";
import { Exp3SphorSection } from "@/features/research/sections/Exp3SphorSection";
import { Exp4NovelSection } from "@/features/research/sections/Exp4NovelSection";
import { Exp5TwoNovelSection } from "@/features/research/sections/Exp5TwoNovelSection";
import { HeroSection } from "@/features/research/sections/HeroSection";
import { IntroSection } from "@/features/research/sections/IntroSection";
import { MethodSection } from "@/features/research/sections/MethodSection";
import { cn } from "@/lib/utils";

/** La tesi: hero, TOC laterale e una sezione per esperimento. */
export function ResearchPage() {
  const { t } = useTranslation();
  const active = useActiveSection();

  return (
    <CiteProvider>
      <div className="relative max-w-7xl mx-auto px-4 md:px-8">
        {/* Sidebar TOC */}
        <aside className="fixed left-0 top-24 bottom-0 w-64 hidden xl:block overflow-y-auto border-r border-border/50 pr-4 pl-6">
          <nav className="space-y-1 pt-6">
            <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground px-3 pb-2">{t("research.toc_title")}</p>
            {SECTIONS.map((s) => (
              <a
                key={s.id}
                href={`#${s.id}`}
                className={cn(
                  "block px-3 py-1.5 text-sm rounded-md transition-colors",
                  active === s.id
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
                )}
              >
                {t(s.labelKey)}
              </a>
            ))}
          </nav>
        </aside>

        {/* Main */}
        <main className="research-content relative xl:ml-64 py-12 max-w-4xl">
          <HeroSection />
          <IntroSection />
          <ContextSection />
          <DatasetSection />
          <MethodSection />
          <Exp1CeSection />
          <Exp2SupconSection />
          <Exp3SphorSection />
          <Exp4NovelSection />
          <Exp5TwoNovelSection />
          <BenchmarkSection />
          <ConclusionsSection />
          <BibliographySection />
        </main>
      </div>
    </CiteProvider>
  );
}
