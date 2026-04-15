// frontend/src/App.tsx

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { DashboardPage } from "@/pages/DashboardPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { ExperimentDetailPage } from "@/pages/ExperimentDetailPage";
import { ClassifyPage } from "@/pages/ClassifyPage";
import { EmbeddingsPage } from "@/pages/EmbeddingsPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/embeddings" element={<EmbeddingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
