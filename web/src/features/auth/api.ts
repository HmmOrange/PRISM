/**
 * Authentication API functions.
 * Following API design standards from CODING_STANDARDS.md
 */

import { apiFetch } from "../../api/client";
import type { AuthResponse, LoginCredentials, RegisterCredentials, User } from "./types";

const AUTH_TOKEN_KEY = "prism_auth_token";

/**
 * Login with username and password.
 * Stores token based on rememberMe preference.
 */
export async function login(credentials: LoginCredentials): Promise<AuthResponse> {
  const response = await apiFetch<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({
      username: credentials.username,
      password: credentials.password,
    }),
  });

  // Store token based on rememberMe preference
  const storage = credentials.rememberMe ? localStorage : sessionStorage;
  storage.setItem(AUTH_TOKEN_KEY, response.access_token);

  return response;
}

/**
 * Register a new user account.
 */
export async function register(credentials: RegisterCredentials): Promise<AuthResponse> {
  const response = await apiFetch<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      username: credentials.username,
      email: credentials.email,
      password: credentials.password,
    }),
  });

  // Auto-login after registration
  sessionStorage.setItem(AUTH_TOKEN_KEY, response.access_token);

  return response;
}

/**
 * Logout the current user.
 * Clears all stored tokens.
 */
export async function logout(): Promise<void> {
  try {
    await apiFetch("/auth/logout", { method: "POST" });
  } finally {
    // Always clear tokens, even if API call fails
    localStorage.removeItem(AUTH_TOKEN_KEY);
    sessionStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

/**
 * Get the current authenticated user.
 * Returns null if not authenticated.
 */
export async function getCurrentUser(): Promise<User | null> {
  const token = getAuthToken();
  if (!token) {
    return null;
  }

  try {
    return await apiFetch<User>("/auth/me");
  } catch {
    // Token is invalid or expired
    clearAuthToken();
    return null;
  }
}

/**
 * Get the stored authentication token.
 */
export function getAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY) ?? sessionStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Clear the stored authentication token.
 */
export function clearAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  sessionStorage.removeItem(AUTH_TOKEN_KEY);
}

/**
 * Check if user is authenticated (has valid token).
 */
export function hasAuthToken(): boolean {
  return getAuthToken() !== null;
}
