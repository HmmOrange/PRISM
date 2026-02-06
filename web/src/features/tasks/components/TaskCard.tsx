import {
  Card,
  CardContent,
  Typography,
  Stack,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Box,
} from "@mui/material";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import TaskStats from "./TaskStats";
import ConfirmDialog from "./ConfirmDialog";
import { deleteTask } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import { useToast } from "../../../components/feedback/ToastProvider";
import { getMetricLabel } from "../../../config/metrics";

interface Props {
  task: TaskListItem;
  onDeleted: (taskId: string) => void;
}

function formatDate(dateString?: string): string {
  if (!dateString) return "";
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function TaskCard({ task, onDeleted }: Props) {
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const { showToast } = useToast();

  const open = Boolean(anchorEl);

  function handleMenuOpen(e: React.MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    setAnchorEl(e.currentTarget);
  }

  function handleMenuClose() {
    setAnchorEl(null);
  }

  function handleDeleteClick(e: React.MouseEvent) {
    e.stopPropagation();
    handleMenuClose();
    setConfirmOpen(true);
  }

  async function handleConfirmDelete() {
    setConfirmOpen(false);
    await deleteTask(task.id);
    onDeleted(task.id);

    showToast({ message: "Task deleted", severity: "success" });
    console.log("Task deleted" );
  }
  return (
    <>
      <Card
        onClick={() => navigate(`/tasks/${task.id}`)}
        sx={{
          height: "100%",
          position: "relative",
          transition: "0.2s",
          "&:hover": { boxShadow: 4 },
        }}
      >
        {/* 3-dots menu */}
        <IconButton
          size="small"
          onClick={handleMenuOpen}
          sx={{ position: "absolute", top: 8, right: 8 }}
        >
          <MoreVertIcon fontSize="small" />
        </IconButton>

        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleMenuClose}
          onClick={(e) => e.stopPropagation()}
        >
          <MenuItem onClick={handleDeleteClick}>Delete</MenuItem>
        </Menu>

        <CardContent>
          <Stack spacing={1.5}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <Typography variant="h6" sx={{ pr: 4 }}>{task.name}</Typography>
            </Box>

            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                minHeight: 40,
              }}
            >
              {task.description || "No description"}
            </Typography>

            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
              <Chip
                label={getMetricLabel(task.metric)}
                size="small"
                color="primary"
                variant="outlined"
              />
              <TaskStats
                test={task.test_queries}
                validation={task.validation_queries}
              />
            </Box>

            {task.created_at && (
              <Typography variant="caption" color="text.secondary">
                Created {formatDate(task.created_at)}
              </Typography>
            )}
          </Stack>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        title="Delete task?"
        description={`This will permanently delete "${task.name}". This action cannot be undone.`}
        confirmText="Delete"
        onConfirm={handleConfirmDelete}
        onCancel={() => setConfirmOpen(false)}
      />
    </> 
  );
}
