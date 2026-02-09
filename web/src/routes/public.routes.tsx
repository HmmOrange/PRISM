/**
 * Public routes configuration.
 * These routes are accessible without authentication.
 * Implements SRS 2.1.2 and 2.2 Navigation requirements.
 */

import type { RouteObject } from "react-router";
import { Navigate } from "react-router-dom";
import { ROUTES } from "../config/routes";

// Auth pages (with guest guard)
import LoginPage from "../features/auth/pages/LoginPage";
import RegisterPage from "../features/auth/pages/RegisterPage";
import { GuestGuard } from "../features/auth";

export const publicRoutes: RouteObject[] = [
  // Auth routes (guest only)
  {
    path: ROUTES.public.login,
    element: (
      <GuestGuard>
        <LoginPage />
      </GuestGuard>
    ),
  },
  {
    path: ROUTES.public.register,
    element: (
      <GuestGuard>
        <RegisterPage />
      </GuestGuard>
    ),
  },

  // Catch-all: redirect unknown routes to login
  {
    path: "*",
    element: <Navigate to={ROUTES.public.login} replace />,
  },
];
