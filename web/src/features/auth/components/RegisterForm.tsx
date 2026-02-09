/**
 * Register form component.
 * Implements SRS 2.1.2 Form Functionality:
 * - Email, username, password fields with client-side validation
 * - Password complexity validation
 *
 * Uses PRISM design-system palette and consistent FormField patterns.
 */

import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Box,
  TextField,
  Button,
  Typography,
  Link,
  Stack,
  Alert,
  CircularProgress,
  Paper,
} from "@mui/material";

import type { RegisterCredentials, RegisterFormErrors } from "../types";
import { ROUTES } from "../../../config/routes";

interface RegisterFormProps {
  onSubmit: (credentials: RegisterCredentials) => Promise<void>;
}

export default function RegisterForm({ onSubmit }: RegisterFormProps) {
  const [credentials, setCredentials] = useState<RegisterCredentials>({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<RegisterFormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  /**
   * Validate email format.
   */
  function isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Validate password complexity.
   * Requires: 8+ chars, 1 uppercase, 1 lowercase, 1 number
   */
  function isValidPassword(password: string): boolean {
    return (
      password.length >= 8 &&
      /[A-Z]/.test(password) &&
      /[a-z]/.test(password) &&
      /\d/.test(password)
    );
  }

  /**
   * Validate form fields.
   */
  function validateForm(): boolean {
    const newErrors: RegisterFormErrors = {};

    if (!credentials.username.trim()) {
      newErrors.username = "Username is required";
    } else if (credentials.username.length < 3) {
      newErrors.username = "Username must be at least 3 characters";
    }

    if (!credentials.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!isValidEmail(credentials.email)) {
      newErrors.email = "Please enter a valid email address";
    }

    if (!credentials.password) {
      newErrors.password = "Password is required";
    } else if (!isValidPassword(credentials.password)) {
      newErrors.password =
        "Password must be 8+ characters with uppercase, lowercase, and number";
    }

    if (!credentials.confirmPassword) {
      newErrors.confirmPassword = "Please confirm your password";
    } else if (credentials.password !== credentials.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
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
          : "Registration failed. Please try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleChange(field: keyof RegisterCredentials, value: string) {
    setCredentials((prev) => ({ ...prev, [field]: value }));

    // Clear field error on change
    if (errors[field as keyof RegisterFormErrors]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  }

  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 3, sm: 4 },
        borderRadius: 3,
        border: "1px solid",
        borderColor: "#BFC9D1",
        bgcolor: "#FFFFFF",
      }}
    >
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <Stack spacing={3}>
          <Box>
            <Typography
              variant="h4"
              fontWeight={700}
              sx={{ color: "#25343F", mb: 0.5 }}
            >
              Create account
            </Typography>
            <Typography sx={{ color: "#25343F", opacity: 0.6 }}>
              Get started with PRISM
            </Typography>
          </Box>

          {errors.general && (
            <Alert severity="error">{errors.general}</Alert>
          )}

          {/* Username */}
          <Box>
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{ mb: 0.75, color: "#25343F" }}
            >
              Username <span style={{ color: "#FF9B51" }}>*</span>
            </Typography>
            <TextField
              placeholder="Choose a username"
              value={credentials.username}
              onChange={(e) => handleChange("username", e.target.value)}
              error={!!errors.username}
              helperText={errors.username}
              fullWidth
              autoComplete="username"
              autoFocus
            />
          </Box>

          {/* Email */}
          <Box>
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{ mb: 0.75, color: "#25343F" }}
            >
              Email <span style={{ color: "#FF9B51" }}>*</span>
            </Typography>
            <TextField
              placeholder="you@example.com"
              type="email"
              value={credentials.email}
              onChange={(e) => handleChange("email", e.target.value)}
              error={!!errors.email}
              helperText={errors.email}
              fullWidth
              autoComplete="email"
            />
          </Box>

          {/* Password */}
          <Box>
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{ mb: 0.75, color: "#25343F" }}
            >
              Password <span style={{ color: "#FF9B51" }}>*</span>
            </Typography>
            <TextField
              placeholder="Create a strong password"
              type="password"
              value={credentials.password}
              onChange={(e) => handleChange("password", e.target.value)}
              error={!!errors.password}
              helperText={
                errors.password ||
                "8+ characters, uppercase, lowercase, number"
              }
              fullWidth
              autoComplete="new-password"
            />
          </Box>

          {/* Confirm Password */}
          <Box>
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{ mb: 0.75, color: "#25343F" }}
            >
              Confirm Password <span style={{ color: "#FF9B51" }}>*</span>
            </Typography>
            <TextField
              placeholder="Re-enter your password"
              type="password"
              value={credentials.confirmPassword}
              onChange={(e) => handleChange("confirmPassword", e.target.value)}
              error={!!errors.confirmPassword}
              helperText={errors.confirmPassword}
              fullWidth
              autoComplete="new-password"
            />
          </Box>

          <Button
            type="submit"
            variant="contained"
            size="large"
            fullWidth
            disabled={isSubmitting}
            sx={{
              bgcolor: "#FF9B51",
              color: "#FFFFFF",
              fontWeight: 600,
              py: 1.4,
              "&:hover": { bgcolor: "#E8863A" },
            }}
          >
            {isSubmitting ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              "Create Account"
            )}
          </Button>

          <Typography variant="body2" align="center" sx={{ color: "#25343F", opacity: 0.6 }}>
            Already have an account?{" "}
            <Link
              component={RouterLink}
              to={ROUTES.public.login}
              sx={{ color: "#FF9B51", fontWeight: 600, textDecoration: "none", "&:hover": { textDecoration: "underline" } }}
            >
              Sign in
            </Link>
          </Typography>
        </Stack>
      </Box>
    </Paper>
  );
}
