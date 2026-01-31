import { API_CONFIG } from "../config/api";

/**
 * JSON API client
 * - Automatically sets JSON headers
 * - Parses JSON responses
 * - Throws meaningful errors
 */
export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_CONFIG.baseUrl}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `API error ${res.status}: ${text || res.statusText}`
    );
  }

  // Make sure to handle 204 No Content responses
  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

/**
 * Raw fetch helper for non-JSON requests (e.g. presigned uploads)
 * - Does NOT set headers
 * - Does NOT parse JSON
 */
export async function rawFetch(
  url: string,
  options: RequestInit
): Promise<Response> {
  return fetch(url, options);
}
