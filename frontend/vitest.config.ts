import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/lib/__tests__/setup.ts"],
    coverage: {
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.d.ts",
        "src/**/__tests__/**",
        "src/components/ui/**",
        "src/locales/**",
        "src/lib/i18n.ts",
        "src/main.tsx",
        "src/vite-env.d.ts",
      ],
    },
  },
});
