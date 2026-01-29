import { createTheme } from "@mui/material/styles";

/**
 * Global MUI theme
 * - Keep this file declarative
 * - No component logic here
 * - Extend gradually (palette → typography → components)
 */

export const theme = createTheme({
  palette: {
    mode: "light",

    primary: {
      main: "#1976d2", // MUI default blue, safe starting point
    },

    secondary: {
      main: "#9c27b0",
    },

    background: {
      default: "#f5f6f8",
      paper: "#ffffff",
    },
  },

  typography: {
    fontFamily: [
      "Inter",
      "-apple-system",
      "BlinkMacSystemFont",
      '"Segoe UI"',
      "Roboto",
      '"Helvetica Neue"',
      "Arial",
      "sans-serif",
    ].join(","),

    h1: {
      fontSize: "2.25rem",
      fontWeight: 600,
    },
    h2: {
      fontSize: "1.75rem",
      fontWeight: 600,
    },
    h3: {
      fontSize: "1.5rem",
      fontWeight: 600,
    },
    h4: {
      fontSize: "1.25rem",
      fontWeight: 600,
    },
    body1: {
      fontSize: "0.95rem",
    },
    body2: {
      fontSize: "0.875rem",
    },
  },

  shape: {
    borderRadius: 8,
  },

  components: {
    MuiButton: {
      defaultProps: {
        disableElevation: true,
      },
    },

    MuiTextField: {
      defaultProps: {
        size: "small",
        variant: "outlined",
      },
    },

    MuiContainer: {
      defaultProps: {
        maxWidth: "lg",
      },
    },
  },
});
