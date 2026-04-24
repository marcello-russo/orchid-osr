// frontend/src/App.tsx

import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { DashboardPage } from "@/pages/DashboardPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { Skeleton } from "@/components/ui/skeleton";

const ExperimentDetailPage = lazy(() =>
  import("@/pages/ExperimentDetailPage").then((m) => ({ default: m.ExperimentDetailPage }))
);
const ClassifyPage = lazy(() =>
  import("@/pages/ClassifyPage").then((m) => ({ default: m.ClassifyPage }))
);
const EmbeddingsPage = lazy(() =>
  import("@/pages/EmbeddingsPage").then((m) => ({ default: m.EmbeddingsPage }))
);

function PageFallback() {
  return (
    <div className="space-y-4 animate-in fade-in duration-300">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-64 w-full rounded-lg" />
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route
            path="/experiments/:id"
            element={<Suspense fallback={<PageFallback />}><ExperimentDetailPage /></Suspense>}
          />
          <Route
            path="/classify"
            element={<Suspense fallback={<PageFallback />}><ClassifyPage /></Suspense>}
          />
          <Route
            path="/embeddings"
            element={<Suspense fallback={<PageFallback />}><EmbeddingsPage /></Suspense>}
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
