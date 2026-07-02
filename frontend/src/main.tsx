// frontend/src/main.tsx

// THREE.Clock deprecato in three@0.184, sostituito con Timer
import * as THREE from "three";
(THREE as any).Clock = THREE.Timer;

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryProvider } from "@/lib/queryClient";
import { useAppStore } from "@/stores/useAppStore";
import App from "./App";
import "./index.css";
import "./lib/i18n";

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
