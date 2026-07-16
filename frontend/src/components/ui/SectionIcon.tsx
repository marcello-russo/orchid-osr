import { cn } from "@/lib/utils";

export function SectionIcon({ icon: Icon, className }: { icon: React.ComponentType<{ className?: string }>; className?: string }) {
  return (
    <div className={cn("size-10 md:size-12 rounded-2xl flex items-center justify-center shrink-0", className)}>
      <Icon className="size-5 md:size-6" />
    </div>
  );
}
