// frontend/src/lib/api.ts

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function fetchApi<T>(path: string, opts?: { method?: string; body?: BodyInit }): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: opts?.method ?? "GET",
    body: opts?.body,
  });
  if (!res.ok) {
    const body = await res.text();
    const snippet = body.length > 200 ? body.slice(0, 200) + "..." : body;
    throw new Error(`API ${res.status}: ${snippet}`);
  }
  return res.json();
}

export type ProgressCallback = (progress: number) => void;

function uploadXHR<T>(
  path: string,
  formData: FormData,
  timeoutMs: number,
  onProgress?: ProgressCallback
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${path}`);
    xhr.timeout = timeoutMs;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid JSON response"));
        }
      } else {
        reject(new Error(`API ${xhr.status}: ${xhr.responseText}`));
      }
    };

    xhr.onerror = () => reject(new Error("Network error — try again"));
    xhr.ontimeout = () => reject(new Error("Request timed out — the server may be busy"));
    xhr.send(formData);
  });
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

export async function classifyUpload(
  file: File,
  modelId?: string,
  onProgress?: ProgressCallback
) {
  const form = new FormData();
  form.append("file", file);
  if (modelId) form.append("model_id", modelId);
  return uploadXHR<import("./types").ClassifyUploadResponse>(
    "/classify/upload",
    form,
    60_000,
    onProgress
  );
}

export async function explainImage(
  file: File,
  modelId?: string,
  onProgress?: ProgressCallback
) {
  const form = new FormData();
  form.append("file", file);
  if (modelId) form.append("model_id", modelId);
  return uploadXHR<import("./types").ExplainResponse>(
    "/classify/explain",
    form,
    300_000,
    onProgress
  );
}

export async function fetchModels() {
  return fetchApi<{ models: import("./types").ModelOption[] }>("/classify/models");
}

export async function fetchCacheStatus() {
  return fetchApi<import("./types").CacheStatus>("/classify/cache");
}

export async function downloadAllModels() {
  return fetchApi<{ status: string }>("/classify/cache/download");
}

export async function downloadModel(id: string) {
  return fetchApi<{ status: string; model_id: string }>(`/classify/cache/download/${encodeURIComponent(id)}`);
}

export async function deleteCachedModel(id: string) {
  return fetchApi<{ status: string; model_id: string }>(`/classify/cache/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function clearModelCache() {
  return fetchApi<{ status: string; files_removed: number }>("/classify/cache", { method: "DELETE" });
}

export async function fetchSpecies() {
  return fetchApi<import("./types").SpeciesListResponse>("/api/species");
}

export async function fetchDatasetClasses() {
  return fetchApi<import("./types").DatasetClassesResponse>("/api/dataset/classes");
}

export async function fetchComparison() {
  return fetchApi<import("./types").ModelComparisonResponse>("/api/comparison");
}

export async function fetchEmbeddings(experiment?: string, method: string = "tsne_3d") {
  const params = new URLSearchParams({ method });
  if (experiment) params.set("experiment", experiment);
  return fetchApi<import("./types").EmbeddingResponse>(`/api/embeddings?${params}`);
}

export async function fetchExpComparison() {
  return fetchApi<import("./types").ExperimentComparisonResponse>("/api/comparison/experiments");
}

export async function fetchExperimentComparison() {
  return fetchApi<import("./types").ExperimentComparisonResponse>("/api/comparison/experiments");
}

export async function fetchComparisonFold(modelVariant: string, fold: number) {
  return fetchApi<import("./types").FoldData>(
    `/api/comparison/${modelVariant}/fold/${fold}`
  );
}

export async function fetchClassImages(className: string, split: string = "train", limit: number = 12, offset: number = 0) {
  return fetchApi<import("./types").ClassImagesResponse>(
    `/api/dataset/classes/${encodeURIComponent(className)}/images?split=${split}&limit=${limit}&offset=${offset}`
  );
}

export async function fetchNcdResults(modelVariant: string, exp: number = 8) {
  return fetchApi<import("./types").NcdResults>(`/api/ncd/${modelVariant}/results?exp=${exp}`);
}

export function fetchNcdFigure(modelVariant: string, figureName: string, exp: number = 8) {
  return `/api/ncd/${modelVariant}/figure/${figureName}?exp=${exp}`;
}

export async function fetchNcdComparison() {
  return fetchApi<import("./types").NcdComparison>("/api/ncd/comparison");
}
