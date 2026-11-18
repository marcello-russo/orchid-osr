// frontend/src/features/explorer/detail/LogsPanel.tsx

import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { fetchExperimentLog } from "@/lib/api";

export function LogsPanel({ experimentId }: { experimentId: string }) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useQuery({
    queryKey: ["log", experimentId],
    queryFn: () => fetchExperimentLog(experimentId),
    staleTime: 60 * 1000,
  });

  if (isLoading) return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  if (error) return <p className="text-sm text-muted-foreground text-center py-8">{t("experiment_detail.log_unavailable")}</p>;
  if (!data?.available) return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <p className="text-sm">{t("experiment_detail.log_unavailable")}</p>
    </div>
  );

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-2">
        <p className="text-xs text-muted-foreground">{t("experiment_detail.log_size")} {(data.size / 1024).toFixed(1)} KB</p>
        <button
          onClick={() => navigator.clipboard.writeText(data.log ?? "")}
          className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80"
        >
          {t("experiment_detail.log_copy")}
        </button>
      </div>
      <pre className="text-xs leading-relaxed bg-black/5 dark:bg-white/5 p-4 rounded-xl overflow-auto max-h-[600px] font-mono">
        {data.log}
      </pre>
    </div>
  );
}
