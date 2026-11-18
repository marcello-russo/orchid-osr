// frontend/src/features/research/components/Prose.tsx

import { useState } from "react";
import { Lightbox } from "@/components/ui/Lightbox";
import { cn } from "@/lib/utils";

/** Primitive di prosa condivise dalle sezioni della tesi. */

export function Figure({ src, alt, caption, className }: { src: string; alt: string; caption?: string; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <figure className="my-8">
      <div className="flex justify-center">
        <img
          src={src} alt={alt}
          className={cn("rounded-xl border border-border cursor-pointer hover:opacity-90 transition-opacity", className || "max-w-full")}
          onClick={() => setOpen(true)}
        />
      </div>
      {caption && <figcaption className="text-sm text-muted-foreground text-center mt-2 italic">{caption}</figcaption>}
      {open && <Lightbox src={src} alt={alt} caption={caption} onClose={() => setOpen(false)} />}
    </figure>
  );
}

export function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 mb-16">
      <h2 className="text-2xl font-bold tracking-tight mb-6 pb-2 border-b border-border/50">{title}</h2>
      {children}
    </section>
  );
}

export function SubSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold tracking-tight mb-3">{title}</h3>
      {children}
    </div>
  );
}

export function P({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground leading-relaxed mb-3">{children}</p>;
}

