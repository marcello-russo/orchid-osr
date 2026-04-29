import { useTranslation } from "react-i18next";
import { Box } from "lucide-react";

export function EmbeddingsPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <Box className="h-16 w-16 text-muted-foreground" />
      <h1 className="mt-4 text-2xl font-semibold text-primary">🧬 {t("embeddings.title")}</h1>
      <p className="mt-2 text-muted-foreground">
        {t("embeddings.description")}
      </p>
    </div>
  );
}
