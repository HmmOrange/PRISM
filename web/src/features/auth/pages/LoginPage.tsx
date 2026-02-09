/**
 * Login Page.
 * Implements SRS 2.1.2 Login/Register Interface
 */

import { useNavigate, useLocation } from "react-router-dom";

import { useAuth, AuthLayout, LoginForm } from "../index";
import type { LoginCredentials } from "../types";
import { ROUTES } from "../../../config/routes";

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();

  async function handleLogin(credentials: LoginCredentials) {
    await login(credentials);
    // Redirect back to where the user was trying to go, or dashboard
    const from =
      (location.state as { from?: { pathname: string } })?.from?.pathname ||
      ROUTES.authed.dashboard;
    navigate(from, { replace: true });
  }

  return (
    <AuthLayout>
      <LoginForm onSubmit={handleLogin} />
    </AuthLayout>
  );
}
