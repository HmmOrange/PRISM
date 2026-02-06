/**
 * Login Page.
 * Implements SRS 2.1.2 Login/Register Interface
 */

import { useNavigate } from "react-router-dom";

import { useAuth, AuthLayout, LoginForm } from "../index";
import type { LoginCredentials } from "../types";
import { ROUTES } from "../../../config/routes";

export default function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();

  async function handleLogin(credentials: LoginCredentials) {
    await login(credentials);
    navigate(ROUTES.public.home, { replace: true });
  }

  return (
    <AuthLayout>
      <LoginForm onSubmit={handleLogin} />
    </AuthLayout>
  );
}
