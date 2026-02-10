import {
  Card,
  CardContent,
  Typography,
  Stack,
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
import { Tag } from "../../../components";

interface Props {
  task: TaskListItem;
  onDeleted: (taskId: string) => void;
}

const MAX_VISIBLE_TAGS = 3;

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
        variant="outlined"
        sx={{
          height: "100%",
          position: "relative",
          cursor: "pointer",
          transition: "0.2s",
          "&:hover": {
            boxShadow: 3,
          },
          minWidth: 0,
          overflow: "hidden",
        }}
      >
        {/* 3-dots menu */}
        <IconButton
          size="small"
          onClick={handleMenuOpen}
          sx={{ position: "absolute", top: 8, right: 8, zIndex: 1 }}
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

        <CardContent sx={{ pr: 5, overflow: "hidden" }}>
          <Stack spacing={1.5} sx={{ minWidth: 0 }}>
            {/* Title and Description */}
            <Box sx={{ minWidth: 0, width: "100%" }}>
              <Typography
                variant="h6"
                sx={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  width: "100%",
                }}
              >
                {task.name}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{
                  mt: 0.5,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  display: "-webkit-box",
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: "vertical",
                  width: "100%",
                }}
              >
                {task.description || "No description"}
              </Typography>
            </Box>

            {/* Pipeline Tags */}
            {task.pipeline_tags && task.pipeline_tags.length > 0 && (
              <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap", alignItems: "center" }}>
                {task.pipeline_tags.slice(0, MAX_VISIBLE_TAGS).map((tag) => (
                  <Tag key={tag} label={tag} size="small" variant="category" />
                ))}
                {task.pipeline_tags.length > MAX_VISIBLE_TAGS && (
                  <Tag
                    label={`+${task.pipeline_tags.length - MAX_VISIBLE_TAGS}`}
                    size="small"
                    variant="default"
                  />
                )}
              </Box>
            )}

            {/* Stats */}
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
