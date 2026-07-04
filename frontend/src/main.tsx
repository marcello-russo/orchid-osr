// frontend/src/main.tsx

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryProvider } from "@/lib/queryClient";
import { useAppStore } from "@/stores/useAppStore";
import App from "./App";
import "./index.css";
import "./lib/i18n";

// Suppress harmless THREE.Clock deprecation warning from @react-three/fiber
const _origWarn = console.warn;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
console.warn = (...args: any[]) => {
  if (typeof args[0] === "string" && args[0].includes("THREE.Clock")) return;
  _origWarn(...args);
};

// Apply theme immediately to prevent flash
const initialTheme = useAppStore.getState().theme;
document.documentElement.classList.toggle("dark", initialTheme === "dark");

window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled promise rejection:", event.reason);
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryProvider>
      <App />
    </QueryProvider>
  </StrictMode>
);
