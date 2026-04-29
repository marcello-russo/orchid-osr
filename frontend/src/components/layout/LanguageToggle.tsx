import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";

export function LanguageToggle() {
  const { i18n } = useTranslation();

  const toggleLanguage = () => {
    const nextLang = i18n.language.startsWith("en") ? "it" : "en";
    i18n.changeLanguage(nextLang);
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggleLanguage}
      className="gap-2 px-3 text-muted-foreground hover:text-primary transition-all font-mono font-bold text-xs"
      title="Switch Language"
    >
      <Languages className="size-4" />
      <span className="uppercase">{i18n.language.startsWith("en") ? "en" : "it"}</span>
    </Button>
  );
}
