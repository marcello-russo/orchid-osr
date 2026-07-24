import { useCiteContext } from "@/features/research/components/CiteContext";
import { CitationPopover } from "@/features/research/components/CitationPopover";
import { PAPER_MAP } from "@/features/research/data/papers";

export function Cite({ id }: { id: string }) {
  const { activeCite, setActiveCite } = useCiteContext();
  const paper = PAPER_MAP[id];
  const isOpen = activeCite === id;

  if (!paper) {
    return <sup className="text-muted-foreground text-[10px]">[{id}]</sup>;
  }

  return (
    <span className="relative inline-block">
      <sup
        id={`cite-${paper.id}`}
        className="text-primary text-[10px] font-medium cursor-pointer hover:underline"
        onClick={() => setActiveCite(isOpen ? null : id)}
      >
        [{paper.shortAuthors}, {paper.year}]
      </sup>
      {isOpen && <CitationPopover paper={paper} onClose={() => setActiveCite(null)} />}
    </span>
  );
}
