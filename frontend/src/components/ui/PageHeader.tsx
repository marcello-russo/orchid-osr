import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function PageHeader({
  icon: Icon,
  title,
  subtitle,
  className,
}: {
  icon: LucideIcon;
  title: string;
  subtitle?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-4", className)}>
      <div className="size-10 md:size-12 rounded-2xl flex items-center justify-center shrink-0">
        <Icon className="size-5 md:size-6" />
      </div>
      <div className="min-w-0">
        <h1 className="text-2xl md:text-4xl font-black tracking-tight">{title}</h1>
        {subtitle && <p className="text-muted-foreground text-sm mt-1">{subtitle}</p>}
      </div>
    </div>
  );
}
