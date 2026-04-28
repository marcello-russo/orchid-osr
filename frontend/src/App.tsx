// frontend/src/App.tsx

import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { ErrorBoundary } from "@/components/ErrorBoundary";
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
const DatasetPage = lazy(() =>
  import("@/pages/DatasetPage").then((m) => ({ default: m.DatasetPage }))
);
const AboutPage = lazy(() =>
  import("@/pages/AboutPage").then((m) => ({ default: m.AboutPage }))
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
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ExperimentDetailPage /></Suspense></ErrorBoundary>}
          />
          <Route
            path="/classify"
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ClassifyPage /></Suspense></ErrorBoundary>}
          />
          <Route
            path="/embeddings"
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><EmbeddingsPage /></Suspense></ErrorBoundary>}
          />
          <Route
            path="/dataset"
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><DatasetPage /></Suspense></ErrorBoundary>}
          />
          <Route
            path="/about"
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><AboutPage /></Suspense></ErrorBoundary>}
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
