/**
 * Authentication Guard component.
 * Implements guard clauses as per SRS 2.1.1
 * 
 * Protects routes by redirecting unauthenticated users to login.
 */

import { Navigate, useLocation } from "react-router-dom";
import { Box, CircularProgress } from "@mui/material";

import { useAuth } from "../context/AuthContext";
import { ROUTES } from "../../../config/routes";

interface AuthGuardProps {
  children: React.ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking auth state
  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return (
      <Navigate
        to={ROUTES.public.login}
        state={{ from: location }}
        replace
      />
    );
  }

  return <>{children}</>;
}
