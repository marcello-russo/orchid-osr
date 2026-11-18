// frontend/src/features/research/sections/BibliographySection.tsx
import { useTranslation } from "react-i18next";
import { Section } from "@/features/research/components/Prose";
import { PAPERS } from "@/features/research/data/papers";

export function BibliographySection() {
  const { t } = useTranslation();
  return (
    <Section id="bibliografia" title={t("research.sect_bibliografia")}>
      <div className="space-y-3 text-sm">
        {PAPERS.map((p) => (
          <div key={p.id} className="flex gap-2">
            <span className="text-primary font-mono text-xs shrink-0 mt-0.5">[{p.shortAuthors}, {p.year}]</span>
            <div>
              <span className="font-medium text-foreground">{p.authors} ({p.year}).</span>{" "}
              <em>{p.title}.</em>{" "}
              {p.venue && <span>{p.venue}. </span>}
              {p.url && (
                <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                  [Link]
                </a>
              )}
              {p.doi && (
                <a href={`https://doi.org/${p.doi}`} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                  [DOI]
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}
