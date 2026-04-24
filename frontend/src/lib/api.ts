// frontend/src/lib/api.ts

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

async function uploadApi<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function fetchExperiments() {
  return fetchApi<import("./types").ExperimentSummary[]>("/api/experiments");
}

export async function fetchExperiment(id: string) {
  return fetchApi<import("./types").ExperimentDetail>(`/api/experiments/${id}`);
}

export async function fetchFold(experimentId: string, foldNumber: number) {
  return fetchApi<import("./types").FoldData>(
    `/api/experiments/${experimentId}/folds/${foldNumber}`
  );
}

export async function classifyUpload(file: File, modelId?: string) {
  const form = new FormData();
  form.append("file", file);
  if (modelId) form.append("model_id", modelId);
  return uploadApi<import("./types").ClassifyUploadResponse>(
    "/classify/upload",
    form
  );
}

export async function explainImage(file: File, modelId?: string) {
  const form = new FormData();
  form.append("file", file);
  if (modelId) form.append("model_id", modelId);
  return uploadApi<import("./types").ExplainResponse>(
    "/classify/explain",
    form
  );
}

export async function fetchModels() {
  return fetchApi<{ models: import("./types").ModelOption[] }>("/classify/models");
}

export async function fetchSpecies() {
  return fetchApi<import("./types").SpeciesListResponse>("/api/species");
}
