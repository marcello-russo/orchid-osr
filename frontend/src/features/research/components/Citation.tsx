import { PAPER_MAP } from "@/features/research/data/papers";

export function Cite({ id }: { id: string }) {
  const paper = PAPER_MAP[id];
  if (!paper) return <sup className="text-muted-foreground text-[10px]">[{id}]</sup>;
  const label = `${paper.authors}, ${paper.year}`;
  const content = <sup className="text-primary text-[10px] font-medium cursor-help hover:underline">[{label}]</sup>;
  if (paper.url) {
    return <a href={paper.url} target="_blank" rel="noopener noreferrer" className="no-underline">{content}</a>;
  }
  if (paper.doi) {
    return <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noopener noreferrer" className="no-underline">{content}</a>;
  }
  return content;
}
