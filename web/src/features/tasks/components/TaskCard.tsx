import {
  Card,
  CardContent,
  Typography,
  Stack,
  Chip,
  IconButton,
  Menu,
  MenuItem,
} from "@mui/material";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import TaskStats from "./TaskStats";
import ConfirmDialog from "./ConfirmDialog";
import { deleteTask } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import { useToast } from "../../../components/feedback/ToastProvider";

interface Props {
  task: TaskListItem;
  onDeleted: (taskId: string) => void;
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
            <Typography variant="h6">{task.name}</Typography>

            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {task.description}
            </Typography>

            <Chip
              label={task.metric}
              size="small"
              sx={{ width: "fit-content" }}
            />

            <TaskStats
              test={task.test_queries}
              validation={task.validation_queries}
            />
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
