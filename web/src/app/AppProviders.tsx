/**
 * Application Providers.
 * Wraps the app with necessary context providers.
 */

import { CssBaseline, ThemeProvider } from "@mui/material";
import type { ReactNode } from "react";
import { theme } from "../styles/theme";
import { ToastProvider } from "../components/feedback/ToastProvider";
import { AuthProvider } from "../features/auth";

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <ToastProvider>   
          {children}
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
