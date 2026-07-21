// frontend/src/App.tsx

import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "sonner";
import { DashboardPage } from "@/pages/DashboardPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { Skeleton } from "@/components/ui/skeleton";
import AuthPage from "@/pages/AuthPage";
import ProtectedRoute from "@/components/auth/ProtectedRoute";

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
const ResearchPage = lazy(() =>
  import("@/pages/ResearchPage").then((m) => ({ default: m.ResearchPage }))
);
const UserManagementPage = lazy(() =>
  import("@/pages/UserManagementPage").then((m) => ({ default: m.UserManagementPage }))
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
      <Toaster position="top-right" richColors />
      <Routes>
        <Route path="/auth" element={<AuthPage />} />

        <Route element={<Layout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route
            path="/experiments/:id"
            element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ExperimentDetailPage /></Suspense></ErrorBoundary>}
          />

          <Route element={<ProtectedRoute role="user" />}>
            <Route
              path="/classify"
              element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ClassifyPage /></Suspense></ErrorBoundary>}
            />
            <Route
              path="/embeddings"
              element={<ErrorBoundary><Suspense fallback={<PageFallback />}><EmbeddingsPage /></Suspense></ErrorBoundary>}
            />
            <Route
              path="/research"
              element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ResearchPage /></Suspense></ErrorBoundary>}
            />
          </Route>

          <Route element={<ProtectedRoute role="admin" />}>
            <Route
              path="/dataset"
              element={<ErrorBoundary><Suspense fallback={<PageFallback />}><DatasetPage /></Suspense></ErrorBoundary>}
            />
            <Route
              path="/admin/users"
              element={<ErrorBoundary><Suspense fallback={<PageFallback />}><UserManagementPage /></Suspense></ErrorBoundary>}
            />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
