import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Il chunk più grosso è three.js (~910 kB, caricato solo dalla pagina
    // embeddings via React.lazy insieme a Scatter3D, quindi fuori dal bundle
    // iniziale che resta ~115 kB). La soglia è 1000 kB, cioè un tetto: sotto
    // c'è solo questa eccezione nota, sopra c'è qualcosa che è cresciuto senza
    // che ce ne accorgessimo — e il warning deve tornare a dire qualcosa.
    chunkSizeWarningLimit: 1000,
  },
  server: {
    port: 5175,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/classify": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/auth": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
