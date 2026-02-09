import { API_CONFIG } from "../config/api";

const AUTH_TOKEN_KEY = "prism_auth_token";

/**
 * Retrieve stored auth token (localStorage or sessionStorage).
 */
function getAuthToken(): string | null {
  return (
    localStorage.getItem(AUTH_TOKEN_KEY) ??
    sessionStorage.getItem(AUTH_TOKEN_KEY)
  );
}

/**
 * JSON API client
 * - Automatically sets JSON headers
 * - Attaches Bearer token when available
 * - Parses JSON responses
 * - Throws meaningful errors
 */
export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  
  // Normalize headers to a plain object
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
  };

  // Only set JSON header when body is NOT FormData
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  // Attach auth token if available
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_CONFIG.baseUrl}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `API error ${res.status}: ${text || res.statusText}`
    );
  }

  // Handle 204 No Content
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
