import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { Compass } from "lucide-react";

export function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <div className="py-24 text-center space-y-3 animate-in fade-in duration-300">
      <div className="size-14 rounded-2xl bg-muted/50 flex items-center justify-center text-muted-foreground mx-auto">
        <Compass className="size-7" />
      </div>
      <p className="text-5xl font-black tracking-tight">404</p>
      <p className="text-lg font-semibold">{t("common.not_found_title")}</p>
      <p className="text-sm text-muted-foreground">{t("common.not_found_text")}</p>
      <Link to="/" className="inline-block pt-2 text-sm text-primary hover:underline">
        {t("common.go_home")}
      </Link>
    </div>
  );
}
