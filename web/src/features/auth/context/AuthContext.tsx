/**
 * Authentication context provider.
 * Implements secure session management as per SRS 2.1.3
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";

import type { User, LoginCredentials, RegisterCredentials, AuthState } from "../types";
import * as authApi from "../api";

interface AuthContextValue extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (credentials: RegisterCredentials) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = user !== null;

  /**
   * Initialize auth state on mount.
   * Checks for existing valid session.
   */
  useEffect(() => {
    async function initAuth() {
      try {
        if (authApi.hasAuthToken()) {
          const currentUser = await authApi.getCurrentUser();
          setUser(currentUser);
        }
      } catch (error) {
        console.error("Auth initialization failed:", error);
        authApi.clearAuthToken();
      } finally {
        setIsLoading(false);
      }
    }

    initAuth();
  }, []);

  /**
   * Login with credentials.
   * Throws on failure for form error handling.
   */
  const login = useCallback(async (credentials: LoginCredentials) => {
    const response = await authApi.login(credentials);
    setUser(response.user);
  }, []);

  /**
   * Register new account.
   * Throws on failure for form error handling.
   */
  const register = useCallback(async (credentials: RegisterCredentials) => {
    const response = await authApi.register(credentials);
    setUser(response.user);
  }, []);

  /**
   * Logout current user.
   */
  const logout = useCallback(async () => {
    await authApi.logout();
    setUser(null);
  }, []);

  /**
   * Refresh user data from server.
   */
  const refreshUser = useCallback(async () => {
    const currentUser = await authApi.getCurrentUser();
    setUser(currentUser);
  }, []);

  const value: AuthContextValue = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Hook to access auth context.
 * Must be used within AuthProvider.
 */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
