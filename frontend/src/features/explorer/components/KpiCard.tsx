// frontend/src/features/explorer/components/KpiCard.tsx

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface KpiCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  className?: string;
  delay?: number;
}

export function KpiCard({ title, value, subtitle, icon: Icon, className, delay = 0 }: KpiCardProps) {
  return (
    <Card
      size="sm"
      className={cn("transition-colors duration-150 animate-in fade-in", className)}
      style={{ animationDelay: `${delay}ms`, animationFillMode: "both" }}
    >
      <CardHeader className="flex flex-row items-center justify-between pb-1">
        <CardTitle className="text-xs font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-xl sm:text-2xl font-bold font-mono">{value}</div>
        {subtitle && (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}
