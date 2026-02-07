/**
 * Universal Tag Component
 * Consistent tag styling across the application.
 * 
 * Variants:
 * - default: Outlined style for unselected items
 * - selected: Filled with accent color for selected items
 * - category: Subtle background for category labels
 */

import { Box, Typography } from "@mui/material";
import { alpha } from "@mui/material/styles";
import CloseIcon from "@mui/icons-material/Close";

import { designTokens } from "../../styles/theme";

export type TagVariant = "default" | "selected" | "category" | "validation" | "test";

interface TagProps {
  label: string;
  variant?: TagVariant;
  onClick?: () => void;
  onRemove?: () => void;
  icon?: React.ReactNode;
  size?: "small" | "medium";
  disabled?: boolean;
}

const variantStyles = {
  default: {
    backgroundColor: "transparent",
    borderColor: designTokens.muted,
    color: designTokens.dark,
    hoverBg: alpha(designTokens.dark, 0.04),
    hoverBorder: designTokens.dark,
  },
  selected: {
    backgroundColor: alpha(designTokens.accent, 0.12),
    borderColor: designTokens.accent,
    color: designTokens.dark,
    hoverBg: alpha(designTokens.accent, 0.18),
    hoverBorder: designTokens.accent,
  },
  category: {
    backgroundColor: alpha(designTokens.dark, 0.06),
    borderColor: "transparent",
    color: designTokens.dark,
    hoverBg: alpha(designTokens.dark, 0.1),
    hoverBorder: "transparent",
  },
  validation: {
    backgroundColor: alpha(designTokens.validation, 0.12),
    borderColor: designTokens.validation,
    color: designTokens.dark,
    hoverBg: alpha(designTokens.validation, 0.18),
    hoverBorder: designTokens.validation,
  },
  test: {
    backgroundColor: alpha(designTokens.test, 0.12),
    borderColor: designTokens.test,
    color: designTokens.dark,
    hoverBg: alpha(designTokens.test, 0.18),
    hoverBorder: designTokens.test,
  },
};

export default function Tag({
  label,
  variant = "default",
  onClick,
  onRemove,
  icon,
  size = "medium",
  disabled = false,
}: TagProps) {
  const styles = variantStyles[variant];
  const isClickable = !!onClick && !disabled;
  const isSmall = size === "small";

  return (
    <Box
      component={isClickable ? "button" : "span"}
      onClick={isClickable ? onClick : undefined}
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.75,
        px: isSmall ? 1 : 1.5,
        py: isSmall ? 0.25 : 0.5,
        borderRadius: 1.5,
        border: "1px solid",
        backgroundColor: styles.backgroundColor,
        borderColor: styles.borderColor,
        color: styles.color,
        fontSize: isSmall ? "0.75rem" : "0.8125rem",
        fontWeight: 500,
        fontFamily: "inherit",
        cursor: isClickable ? "pointer" : "default",
        opacity: disabled ? 0.5 : 1,
        transition: "all 0.15s ease-in-out",
        whiteSpace: "nowrap",
        outline: "none",
        
        "&:hover": isClickable ? {
          backgroundColor: styles.hoverBg,
          borderColor: styles.hoverBorder,
          transform: "translateY(-1px)",
        } : {},
        
        "&:active": isClickable ? {
          transform: "translateY(0)",
        } : {},
        
        "&:focus-visible": {
          boxShadow: `0 0 0 2px ${alpha(designTokens.accent, 0.4)}`,
        },
      }}
    >
      {icon && (
        <Box
          component="span"
          sx={{
            display: "flex",
            alignItems: "center",
            "& > svg": {
              fontSize: isSmall ? "0.875rem" : "1rem",
            },
          }}
        >
          {icon}
        </Box>
      )}
      
      <Typography
        component="span"
        sx={{
          fontSize: "inherit",
          fontWeight: "inherit",
          lineHeight: 1.4,
        }}
      >
        {label}
      </Typography>

      {onRemove && !disabled && (
        <Box
          component="span"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          sx={{
            display: "flex",
            alignItems: "center",
            ml: 0.25,
            p: 0.25,
            borderRadius: "50%",
            cursor: "pointer",
            transition: "all 0.15s ease-in-out",
            "&:hover": {
              backgroundColor: alpha(
                variant === "selected" ? "#FFFFFF" : designTokens.dark,
                0.2
              ),
            },
          }}
        >
          <CloseIcon sx={{ fontSize: isSmall ? "0.75rem" : "0.875rem" }} />
        </Box>
      )}
    </Box>
  );
}
