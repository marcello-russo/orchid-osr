import { AlertTriangle, Info, CheckCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const variants = {
  info: { icon: Info, bg: "bg-blue-500/5 dark:bg-blue-500/10 border-blue-200/50 dark:border-blue-800/30", text: "text-blue-700 dark:text-blue-300", iconColor: "text-blue-500" },
  warning: { icon: AlertTriangle, bg: "bg-amber-500/5 dark:bg-amber-500/10 border-amber-300/50 dark:border-amber-700/30", text: "text-amber-700 dark:text-amber-300", iconColor: "text-amber-500" },
  success: { icon: CheckCircle, bg: "bg-green-500/5 dark:bg-green-500/10 border-green-300/50 dark:border-green-700/30", text: "text-green-700 dark:text-green-300", iconColor: "text-green-500" },
  danger: { icon: XCircle, bg: "bg-red-500/5 dark:bg-red-500/10 border-red-300/50 dark:border-red-700/30", text: "text-red-700 dark:text-red-300", iconColor: "text-red-500" },
};

export function Callout({ variant, title, children }: { variant: keyof typeof variants; title: string; children: React.ReactNode }) {
  const v = variants[variant];
  const Icon = v.icon;
  return (
    <div className={cn("rounded-lg border p-4 my-6", v.bg)}>
      <div className="flex gap-3">
        <Icon className={cn("size-5 shrink-0 mt-0.5", v.iconColor)} />
        <div>
          <p className={cn("font-semibold text-sm", v.text)}>{title}</p>
          <div className={cn("text-sm mt-1 leading-relaxed", v.text)}>{children}</div>
        </div>
      </div>
    </div>
  );
}
