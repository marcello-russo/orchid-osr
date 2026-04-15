// frontend/src/pages/ClassifyPage.tsx

import { ScanSearch } from "lucide-react";

export function ClassifyPage() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <ScanSearch className="h-16 w-16 text-muted-foreground" />
      <h1 className="mt-4 text-2xl font-semibold text-primary">Classify</h1>
      <p className="mt-2 text-muted-foreground">
        Coming soon — image classification UI
      </p>
    </div>
  );
}
