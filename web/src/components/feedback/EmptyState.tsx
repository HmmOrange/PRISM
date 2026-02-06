/**
 * Empty State component.
 * Used when there's no data to display.
 */

import { Box, Typography, Button, Paper } from "@mui/material";
import InboxIcon from "@mui/icons-material/Inbox";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function EmptyState({
  icon,
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 6,
        textAlign: "center",
        bgcolor: "background.default",
      }}
    >
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          mb: 2,
          color: "text.secondary",
        }}
      >
        {icon || <InboxIcon sx={{ fontSize: 64 }} />}
      </Box>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      {description && (
        <Typography variant="body2" color="text.secondary" mb={3}>
          {description}
        </Typography>
      )}
      {action && (
        <Button variant="contained" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </Paper>
  );
}
