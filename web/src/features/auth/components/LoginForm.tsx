/**
 * Login form component.
 * Implements SRS 2.1.2 Form Functionality:
 * - Email/username and password fields with validation
 * - Remember Me checkbox for persistent sessions
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
  FormControlLabel,
  Checkbox,
  Link,
  Stack,
  Alert,
  CircularProgress,
  Paper,
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
              Welcome back
            </Typography>
            <Typography sx={{ color: "#25343F", opacity: 0.6 }}>
              Sign in to your PRISM account
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
              placeholder="Enter your username"
              type="text"
              value={credentials.username}
              onChange={(e) => handleChange("username", e.target.value)}
              error={!!errors.username}
              helperText={errors.username}
              fullWidth
              autoComplete="username"
              autoFocus
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
              placeholder="Enter your password"
              type="password"
              value={credentials.password}
              onChange={(e) => handleChange("password", e.target.value)}
              error={!!errors.password}
              helperText={errors.password}
              fullWidth
              autoComplete="current-password"
            />
          </Box>

          <FormControlLabel
            control={
              <Checkbox
                checked={credentials.rememberMe}
                onChange={(e) => handleChange("rememberMe", e.target.checked)}
                sx={{
                  color: "#BFC9D1",
                  "&.Mui-checked": { color: "#FF9B51" },
                }}
              />
            }
            label={
              <Typography variant="body2" sx={{ color: "#25343F" }}>
                Remember me
              </Typography>
            }
          />

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
            {isSubmitting ? <CircularProgress size={24} color="inherit" /> : "Sign In"}
          </Button>

          <Typography variant="body2" align="center" sx={{ color: "#25343F", opacity: 0.6 }}>
            Don't have an account?{" "}
            <Link
              component={RouterLink}
              to={ROUTES.public.register}
              sx={{ color: "#FF9B51", fontWeight: 600, textDecoration: "none", "&:hover": { textDecoration: "underline" } }}
            >
              Sign up
            </Link>
          </Typography>
        </Stack>
      </Box>
    </Paper>
  );
}
