// frontend/src/pages/EmbeddingsPage.tsx

import { Box } from "lucide-react";

export function EmbeddingsPage() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <Box className="h-16 w-16 text-muted-foreground" />
      <h1 className="mt-4 text-2xl font-semibold text-primary">Embeddings</h1>
      <p className="mt-2 text-muted-foreground">
        Coming soon — Phase 2: Contrastive Learning
      </p>
    </div>
  );
}
