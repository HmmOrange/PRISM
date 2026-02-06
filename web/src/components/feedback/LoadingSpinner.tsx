/**
 * Loading Spinner component.
 * Implements SRS 3.1: All async actions must show loading spinners.
 */

import { Box, CircularProgress, Typography } from "@mui/material";

interface LoadingSpinnerProps {
  message?: string;
  size?: number;
  fullPage?: boolean;
}

export default function LoadingSpinner({
  message,
  size = 40,
  fullPage = false,
}: LoadingSpinnerProps) {
  const content = (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      gap={2}
      p={4}
    >
      <CircularProgress size={size} />
      {message && (
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
      )}
    </Box>
  );

  if (fullPage) {
    return (
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
      >
        {content}
      </Box>
    );
  }

  return content;
}
