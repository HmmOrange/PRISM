/**
 * Error Display component.
 * Implements SRS 3.1: Error handling with clear messaging.
 */

import { Typography, Button, Paper, Alert } from "@mui/material";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";

interface ErrorDisplayProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  variant?: "inline" | "full";
}

export default function ErrorDisplay({
  title = "Something went wrong",
  message,
  onRetry,
  variant = "inline",
}: ErrorDisplayProps) {
  if (variant === "inline") {
    return (
      <Alert
        severity="error"
        action={
          onRetry && (
            <Button color="inherit" size="small" onClick={onRetry}>
              Retry
            </Button>
          )
        }
      >
        {message}
      </Alert>
    );
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 4,
        textAlign: "center",
        borderColor: "error.light",
      }}
    >
      <ErrorOutlineIcon
        sx={{ fontSize: 48, color: "error.main", mb: 2 }}
      />
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        {message}
      </Typography>
      {onRetry && (
        <Button variant="outlined" color="error" onClick={onRetry}>
          Try Again
        </Button>
      )}
    </Paper>
  );
}
