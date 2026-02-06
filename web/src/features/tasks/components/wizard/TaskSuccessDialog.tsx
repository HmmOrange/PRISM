/**
 * Task Creation Success Dialog.
 * Implements SRS 2.3.1 Stage 4 Success State:
 * - Success toast/banner
 * - View Task action
 * - Generate Workflow action (disabled for Phase 1)
 */

import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Stack,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import VisibilityIcon from "@mui/icons-material/Visibility";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

interface TaskSuccessDialogProps {
  open: boolean;
  taskId: string;
  taskName: string;
  onViewTask: () => void;
  onGenerateWorkflow: () => void;
  onClose: () => void;
}

export default function TaskSuccessDialog({
  open,
  taskName,
  onViewTask,
  onGenerateWorkflow,
  onClose,
}: TaskSuccessDialogProps) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ textAlign: "center", pt: 4 }}>
        <CheckCircleIcon
          color="success"
          sx={{ fontSize: 64, mb: 2 }}
        />
        <Typography variant="h5" fontWeight={600}>
          Task Created Successfully!
        </Typography>
      </DialogTitle>

      <DialogContent>
        <Box textAlign="center" py={2}>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            Your task <strong>"{taskName}"</strong> has been created.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            What would you like to do next?
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 3, justifyContent: "center" }}>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} width="100%">
          <Button
            variant="contained"
            startIcon={<VisibilityIcon />}
            onClick={onViewTask}
            fullWidth
          >
            View Task
          </Button>

          <Button
            variant="outlined"
            startIcon={<AutoFixHighIcon />}
            onClick={onGenerateWorkflow}
            fullWidth
            disabled // Disabled for Phase 1 as per SRS
            title="Coming soon - Generate Workflow"
          >
            Generate Workflow
          </Button>
        </Stack>
      </DialogActions>
    </Dialog>
  );
}
