import { AlertTriangle, Info, CheckCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const variants = {
  info: { icon: Info, bg: "bg-blue-500/[0.04] dark:bg-blue-500/[0.08] border-blue-200/40 dark:border-blue-800/30", iconColor: "text-blue-500" },
  warning: { icon: AlertTriangle, bg: "bg-amber-500/[0.04] dark:bg-amber-500/[0.08] border-amber-300/40 dark:border-amber-700/30", iconColor: "text-amber-500" },
  success: { icon: CheckCircle, bg: "bg-green-500/[0.04] dark:bg-green-500/[0.08] border-green-300/40 dark:border-green-700/30", iconColor: "text-green-500" },
  danger: { icon: XCircle, bg: "bg-red-500/[0.04] dark:bg-red-500/[0.08] border-red-300/40 dark:border-red-700/30", iconColor: "text-red-500" },
};

export function Callout({ variant, title, children }: { variant: keyof typeof variants; title: string; children: React.ReactNode }) {
  const v = variants[variant];
  const Icon = v.icon;
  return (
    <div className={cn("rounded-lg border p-4 my-6", v.bg)}>
      <div className="flex gap-3">
        <Icon className={cn("size-5 shrink-0 mt-0.5", v.iconColor)} />
        <div>
          <p className="font-semibold text-sm text-foreground">{title}</p>
          <div className="text-sm mt-1 leading-relaxed text-foreground/80">{children}</div>
        </div>
      </div>
    </div>
  );
}
