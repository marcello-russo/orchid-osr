// frontend/src/lib/api.ts

import { useAppStore } from "../stores/useAppStore";
import type { AuthUser } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

function getAuthHeaders(): Record<string, string> {
  const token = useAppStore.getState().token;
  if (token) return { Authorization: `Bearer ${token}` };
  return {};
}

async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
    ...getAuthHeaders(),
  };
  if (url.startsWith("/auth/login") || url.startsWith("/auth/register")) {
    delete headers.Authorization;
  }
  return fetch(url, { ...options, headers });
}

export async function fetchApi<T>(path: string, opts?: { method?: string; body?: BodyInit }): Promise<T> {
  const res = await apiFetch(`${API_BASE}${path}`, {
    method: opts?.method ?? "GET",
    body: opts?.body,
  });
  if (!res.ok) {
    if (res.status === 401) {
        const { useAppStore } = await import("../stores/useAppStore");
        useAppStore.getState().logout();
        window.location.href = "/auth";
        throw new Error("Sessione scaduta");
    }
    const body = await res.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === "string") detail = parsed.detail;
      else if (parsed.detail?.[0]?.msg) detail = parsed.detail[0].msg;
    } catch {}
    const snippet = detail.length > 200 ? detail.slice(0, 200) + "..." : detail;
    if (res.status >= 400 && res.status < 500) {
        const { toast } = await import("sonner");
        toast.error(`Errore ${res.status}: ${snippet}`);
        throw new Error(`API ${res.status}: ${snippet}`);
    }
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

export async function fetchExperimentLog(experimentId: string) {
  return fetchApi<{ log: string | null; size: number; available: boolean }>(`/api/experiments/${experimentId}/log`);
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

export async function fetchModels() {
  return fetchApi<{ models: import("./types").ModelOption[] }>("/classify/models");
}

export async function fetchCacheStatus() {
  return fetchApi<import("./types").CacheStatus>("/api/admin/cache");
}

export async function downloadAllModels() {
  return fetchApi<{ status: string }>("/api/admin/cache/download", { method: "POST" });
}

export async function downloadModel(id: string) {
  return fetchApi<{ status: string; model_id: string }>(`/api/admin/cache/download/${encodeURIComponent(id)}`, { method: "POST" });
}

export async function deleteCachedModel(id: string) {
  return fetchApi<{ status: string; model_id: string }>(`/api/admin/cache/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function clearModelCache() {
  return fetchApi<{ status: string; files_removed: number }>("/api/admin/cache", { method: "DELETE" });
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

export async function login(username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  const res = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text();
    try {
      const parsed = JSON.parse(body);
      throw new Error(`API ${res.status}: ${parsed.detail || parsed.error || body}`);
    } catch {
      throw new Error(`API ${res.status}: ${body}`);
    }
  }
  return res.json();
}

export async function register(inviteCode: string, username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  return fetchApi("/auth/register", {
    method: "POST",
    body: JSON.stringify({ invite_code: inviteCode, username, password }),
  });
}

export async function fetchInvites(): Promise<{
  invites: Array<{
    code: string;
    role: string;
    used: boolean;
    used_by: string | null;
    created_at: string | null;
  }>;
  users: Array<{ username: string; role: string }>;
}> {
  return fetchApi("/auth/invites");
}

export async function generateInvites(
  count: number,
  role: string
): Promise<{ codes: string[] }> {
  return fetchApi("/auth/invites/generate", {
    method: "POST",
    body: JSON.stringify({ count, role }),
  });
}

export async function fetchMe(): Promise<{ username: string; role: string }> {
  const res = await apiFetch("/auth/me");
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

export async function deleteUser(username: string): Promise<void> {
  await fetchApi(`/auth/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
}
