/**
 * Login form component.
 * Implements SRS 2.1.2 Form Functionality:
 * - Email/username and password fields with validation
 * - Remember Me checkbox for persistent sessions
 */

import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Box,
  TextField,
  Button,
  Typography,
  FormControlLabel,
  Checkbox,
  Link,
  Stack,
  Alert,
  CircularProgress,
} from "@mui/material";

import type { LoginCredentials, LoginFormErrors } from "../types";
import { ROUTES } from "../../../config/routes";

interface LoginFormProps {
  onSubmit: (credentials: LoginCredentials) => Promise<void>;
}

export default function LoginForm({ onSubmit }: LoginFormProps) {
  const [credentials, setCredentials] = useState<LoginCredentials>({
    username: "",
    password: "",
    rememberMe: false,
  });
  const [errors, setErrors] = useState<LoginFormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  /**
   * Validate form fields.
   * Returns true if valid, false otherwise.
   */
  function validateForm(): boolean {
    const newErrors: LoginFormErrors = {};

    if (!credentials.username.trim()) {
      newErrors.username = "Username is required";
    }

    if (!credentials.password) {
      newErrors.password = "Password is required";
    } else if (credentials.password.length < 4) {
      newErrors.password = "Password must be at least 4 characters";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setErrors({});

    try {
      await onSubmit(credentials);
    } catch (error) {
      setErrors({
        general: error instanceof Error
          ? error.message
          : "Login failed. Please check your credentials.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleChange(field: keyof LoginCredentials, value: string | boolean) {
    setCredentials((prev) => ({ ...prev, [field]: value }));

    // Clear field error on change
    if (errors[field as keyof LoginFormErrors]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  }

  return (
    <Box component="form" onSubmit={handleSubmit} noValidate>
      <Stack spacing={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Welcome back
          </Typography>
          <Typography color="text.secondary">
            Sign in to your PRISM account
          </Typography>
        </Box>

        {errors.general && (
          <Alert severity="error">{errors.general}</Alert>
        )}

        <TextField
          label="Username"
          type="text"
          value={credentials.username}
          onChange={(e) => handleChange("username", e.target.value)}
          error={!!errors.username}
          helperText={errors.username}
          fullWidth
          required
          autoComplete="username"
          autoFocus
        />

        <TextField
          label="Password"
          type="password"
          value={credentials.password}
          onChange={(e) => handleChange("password", e.target.value)}
          error={!!errors.password}
          helperText={errors.password}
          fullWidth
          required
          autoComplete="current-password"
        />

        <FormControlLabel
          control={
            <Checkbox
              checked={credentials.rememberMe}
              onChange={(e) => handleChange("rememberMe", e.target.checked)}
              color="primary"
            />
          }
          label="Remember me"
        />

        <Button
          type="submit"
          variant="contained"
          size="large"
          fullWidth
          disabled={isSubmitting}
        >
          {isSubmitting ? <CircularProgress size={24} /> : "Sign In"}
        </Button>

        <Typography variant="body2" align="center">
          Don't have an account?{" "}
          <Link component={RouterLink} to={ROUTES.public.register}>
            Sign up
          </Link>
        </Typography>
      </Stack>
    </Box>
  );
}
