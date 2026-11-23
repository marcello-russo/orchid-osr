// frontend/src/components/ui/InfoTooltip.tsx
import { HelpCircle } from "lucide-react";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "@/components/ui/tooltip";

interface InfoTooltipProps {
  text: string;
}

export function InfoTooltip({ text }: InfoTooltipProps) {
  return (
    <TooltipProvider delay={200}>
      <Tooltip>
        <TooltipTrigger className="inline-flex align-middle ml-1 cursor-help">
          <HelpCircle className="size-3.5 text-muted-foreground/60" />
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs text-balance">
          {text}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
