/**
 * FormField Wrapper Component
 * Standardizes form field layout:
 * - Label above the input
 * - Description/helper text below label (optional)
 * - Error message below input
 * - Required indicator
 */

import { Box, Typography, FormHelperText } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { designTokens } from "../../styles/theme";

interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
  id?: string;
}

export default function FormField({
  label,
  description,
  required = false,
  error,
  children,
  id,
}: FormFieldProps) {
  return (
    <Box sx={{ width: "100%" }}>
      {/* Label */}
      <Typography
        component="label"
        htmlFor={id}
        sx={{
          display: "block",
          fontSize: "0.875rem",
          fontWeight: 600,
          color: designTokens.dark,
          mb: 0.5,
        }}
      >
        {label}
        {required && (
          <Box
            component="span"
            sx={{
              color: designTokens.accent,
              ml: 0.5,
            }}
          >
            *
          </Box>
        )}
      </Typography>

      {/* Description */}
      {description && (
        <Typography
          variant="body2"
          sx={{
            color: alpha(designTokens.dark, 0.6),
            mb: 1,
            fontSize: "0.8125rem",
          }}
        >
          {description}
        </Typography>
      )}

      {/* Input */}
      <Box>{children}</Box>

      {/* Error */}
      {error && (
        <FormHelperText
          error
          sx={{
            mt: 0.5,
            mx: 0,
            fontSize: "0.8125rem",
          }}
        >
          {error}
        </FormHelperText>
      )}
    </Box>
  );
}
