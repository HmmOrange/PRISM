/**
 * Register Page.
 * Implements SRS 2.1.2 Login/Register Interface
 */

import { useNavigate } from "react-router-dom";

import { useAuth, AuthLayout, RegisterForm } from "../index";
import type { RegisterCredentials } from "../types";
import { ROUTES } from "../../../config/routes";

export default function RegisterPage() {
  const navigate = useNavigate();
  const { register } = useAuth();

  async function handleRegister(credentials: RegisterCredentials) {
    await register(credentials);
    navigate(ROUTES.authed.dashboard, { replace: true });
  }

  return (
    <AuthLayout>
      <RegisterForm onSubmit={handleRegister} />
    </AuthLayout>
  );
}
