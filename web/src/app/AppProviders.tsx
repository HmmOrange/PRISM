import { CssBaseline, ThemeProvider } from "@mui/material";
import type { ReactNode } from "react";
import { theme } from "../styles/theme";
import { ToastProvider } from "../components/feedback/ToastProvider";

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ToastProvider>   
        {children}
      </ToastProvider>
    </ThemeProvider>
  );
}
