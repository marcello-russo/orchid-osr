// frontend/src/components/layout/ThemeToggle.tsx

import { Moon, Sun } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAppStore } from "@/stores/useAppStore";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { t } = useTranslation();
  const { theme, toggleTheme } = useAppStore();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      aria-label={t("common.theme_toggle")}
    >
      {theme === "dark" ? (
        <Sun className="h-5 w-5" />
      ) : (
        <Moon className="h-5 w-5" />
      )}
    </Button>
  );
}
