/**
 * Authentication types for the auth module.
 * Following TypeScript strict mode conventions from CODING_STANDARDS.md
 */

export interface User {
  id: string;
  username: string;
  email: string;
  created_at: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterCredentials {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export interface AuthResponse {
  user: User;
  access_token: string;
  token_type: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface LoginFormErrors {
  username?: string;
  password?: string;
  general?: string;
}

export interface RegisterFormErrors {
  username?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  general?: string;
}
