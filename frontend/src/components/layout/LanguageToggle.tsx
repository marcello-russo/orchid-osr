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
      variant="outline"
      size="sm"
      onClick={toggleLanguage}
      className="gap-1.5 px-2.5 border-primary/20 hover:bg-primary/10 text-xs font-bold tracking-wider"
      title="Switch Language"
    >
      <Languages className="size-3.5" />
      <span className="uppercase">{i18n.language.startsWith("en") ? "en" : "it"}</span>
    </Button>
  );
}
