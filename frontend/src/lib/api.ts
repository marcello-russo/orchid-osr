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

export async function fetchSpecies() {
  return fetchApi<import("./types").SpeciesListResponse>("/api/species");
}
