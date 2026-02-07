import { createTheme, alpha } from "@mui/material/styles";

/**
 * Global MUI theme
 * PRISM Design System
 * 
 * Color Palette:
 * - Light: #EAEFEF (backgrounds)
 * - Muted: #BFC9D1 (borders, disabled)
 * - Dark: #25343F (text, primary actions)
 * - Accent: #FF9B51 (CTAs, highlights)
 */

// Design tokens
const colors = {
  light: "#EAEFEF",
  muted: "#BFC9D1",
  dark: "#25343F",
  accent: "#FF9B51",
  white: "#FFFFFF",
  success: "#22C55E",
  error: "#EF4444",
  warning: "#F59E0B",
  info: "#3B82F6",
  // Semantic colors for dataset types
  validation: "#22C55E", // Green - same as success
  test: "#3B82F6", // Blue - same as info
};

export const theme = createTheme({
  palette: {
    mode: "light",

    primary: {
      main: colors.dark,
      light: alpha(colors.dark, 0.7),
      dark: colors.dark,
      contrastText: colors.white,
    },

    secondary: {
      main: colors.accent,
      light: alpha(colors.accent, 0.8),
      dark: "#E8863A",
      contrastText: colors.dark,
    },

    background: {
      default: colors.light,
      paper: colors.white,
    },

    text: {
      primary: colors.dark,
      secondary: alpha(colors.dark, 0.6),
      disabled: colors.muted,
    },

    divider: colors.muted,

    success: {
      main: colors.success,
      light: alpha(colors.success, 0.1),
    },
    error: {
      main: colors.error,
      light: alpha(colors.error, 0.1),
    },
    warning: {
      main: colors.warning,
      light: alpha(colors.warning, 0.1),
    },
    info: {
      main: colors.info,
      light: alpha(colors.info, 0.1),
    },

    action: {
      hover: alpha(colors.dark, 0.04),
      selected: alpha(colors.accent, 0.12),
      focus: alpha(colors.accent, 0.12),
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
      fontWeight: 700,
      color: colors.dark,
    },
    h2: {
      fontSize: "1.75rem",
      fontWeight: 600,
      color: colors.dark,
    },
    h3: {
      fontSize: "1.5rem",
      fontWeight: 600,
      color: colors.dark,
    },
    h4: {
      fontSize: "1.25rem",
      fontWeight: 600,
      color: colors.dark,
    },
    h5: {
      fontSize: "1.125rem",
      fontWeight: 600,
      color: colors.dark,
    },
    h6: {
      fontSize: "1rem",
      fontWeight: 600,
      color: colors.dark,
    },
    body1: {
      fontSize: "0.95rem",
    },
    body2: {
      fontSize: "0.875rem",
    },
    caption: {
      fontSize: "0.75rem",
      color: alpha(colors.dark, 0.6),
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
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 500,
          borderRadius: 6,
          transition: "all 0.2s ease-in-out",
        },
        contained: {
          "&:hover": {
            transform: "translateY(-1px)",
            boxShadow: `0 4px 12px ${alpha(colors.dark, 0.15)}`,
          },
        },
        containedSecondary: {
          color: colors.white,
          "&:hover": {
            backgroundColor: "#E8863A",
          },
        },
        outlined: {
          borderColor: colors.muted,
          "&:hover": {
            borderColor: colors.dark,
            backgroundColor: alpha(colors.dark, 0.04),
          },
        },
      },
    },

    MuiTextField: {
      defaultProps: {
        size: "small",
        variant: "outlined",
      },
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            backgroundColor: colors.white,
            transition: "all 0.2s ease-in-out",
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: colors.dark,
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderColor: colors.accent,
              borderWidth: 2,
            },
          },
          "& .MuiInputLabel-root.Mui-focused": {
            color: colors.dark,
          },
        },
      },
    },

    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
          borderRadius: 6,
          transition: "all 0.15s ease-in-out",
        },
        outlined: {
          borderColor: colors.muted,
          "&:hover": {
            borderColor: colors.dark,
          },
        },
        filled: {
          "&.MuiChip-colorPrimary": {
            backgroundColor: colors.dark,
            color: colors.white,
          },
          "&.MuiChip-colorSecondary": {
            backgroundColor: colors.accent,
            color: colors.white,
          },
        },
      },
    },

    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
        outlined: {
          borderColor: colors.muted,
        },
      },
    },

    MuiContainer: {
      defaultProps: {
        maxWidth: "lg",
      },
    },

    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: colors.white,
          borderBottom: `1px solid ${colors.muted}`,
        },
      },
    },

    MuiMenu: {
      styleOverrides: {
        paper: {
          boxShadow: `0 4px 20px ${alpha(colors.dark, 0.1)}`,
          border: `1px solid ${colors.muted}`,
        },
      },
    },

    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: colors.dark,
          fontSize: "0.75rem",
        },
      },
    },

    MuiStepper: {
      styleOverrides: {
        root: {
          "& .MuiStepIcon-root.Mui-active": {
            color: colors.accent,
          },
          "& .MuiStepIcon-root.Mui-completed": {
            color: colors.success,
          },
        },
      },
    },

    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
        standardSuccess: {
          backgroundColor: alpha(colors.success, 0.1),
          color: colors.dark,
        },
        standardError: {
          backgroundColor: alpha(colors.error, 0.1),
          color: colors.dark,
        },
        standardWarning: {
          backgroundColor: alpha(colors.warning, 0.1),
          color: colors.dark,
        },
        standardInfo: {
          backgroundColor: alpha(colors.info, 0.1),
          color: colors.dark,
        },
      },
    },

    MuiAccordion: {
      styleOverrides: {
        root: {
          "&:before": {
            display: "none",
          },
          boxShadow: "none",
          border: `1px solid ${colors.muted}`,
          borderRadius: "8px !important",
          "&.Mui-expanded": {
            margin: 0,
          },
        },
      },
    },
  },
});

// Export design tokens for use in custom components
export { colors as designTokens };
