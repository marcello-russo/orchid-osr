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
