/**
 * Task List View component.
 * Alternative view for task library - simplified row-based layout.
 */

import {
  Paper,
  IconButton,
  Tooltip,
  Typography,
  Box,
  Stack,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import DeleteIcon from "@mui/icons-material/Delete";
import VisibilityIcon from "@mui/icons-material/Visibility";

import type { TaskListItem } from "../../../types/tasks.types";
import { deleteTask } from "../../../api/tasks.api";
import { useToast } from "../../../components/feedback/ToastProvider";
import { Tag } from "../../../components";

interface TaskListViewProps {
  tasks: TaskListItem[];
  onTaskDeleted: (taskId: string) => void;
}

const MAX_VISIBLE_TAGS = 3;

export default function TaskListView({ tasks, onTaskDeleted }: TaskListViewProps) {
  const navigate = useNavigate();
  const { showToast } = useToast();

  async function handleDelete(taskId: string, taskName: string, e: React.MouseEvent) {
    e.stopPropagation();
    
    if (!confirm(`Delete task "${taskName}"?`)) {
      return;
    }

    try {
      await deleteTask(taskId);
      onTaskDeleted(taskId);
      showToast({ message: `Task "${taskName}" deleted`, severity: "success" });
    } catch (err) {
      showToast({ message: "Failed to delete task", severity: "error" });
    }
  }

  return (
    <Stack spacing={1}>
      {tasks.map((task) => (
        <Paper
          key={task.id}
          variant="outlined"
          onClick={() => navigate(`/tasks/${task.id}`)}
          sx={{
            p: 2,
            cursor: "pointer",
            transition: "0.2s",
            "&:hover": {
              boxShadow: 3,
            },
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "flex-start",
              gap: 2,
            }}
          >
            {/* Left: Name, Description, Tags */}
            <Box sx={{ flex: 1, minWidth: 0, overflow: "hidden", width: "100%" }}>
              {/* Task Name */}
              <Typography
                variant="subtitle1"
                fontWeight={600}
                noWrap
                sx={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "block",
                  width: "100%",
                }}
              >
                {task.name}
              </Typography>

              {/* Description */}
              <Typography
                variant="body2"
                color="text.secondary"
                noWrap
                sx={{
                  mt: 0.25,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "block",
                  width: "100%",
                }}
              >
                {task.description || "No description"}
              </Typography>

              {/* Pipeline Tags */}
              {task.pipeline_tags && task.pipeline_tags.length > 0 && (
                <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", mt: 1 }}>
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
            </Box>

            {/* Middle: Query Stats */}
            <Stack direction="row" spacing={0.5} sx={{ flexShrink: 0 }}>
              <Tag label={`Test ${task.test_queries}`} size="small" variant="test" />
              <Tag label={`Val ${task.validation_queries}`} size="small" variant="validation" />
            </Stack>

            {/* Right: Actions */}
            <Stack direction="row" spacing={0.5} sx={{ flexShrink: 0 }}>
              <Tooltip title="View">
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/tasks/${task.id}`);
                  }}
                >
                  <VisibilityIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Delete">
                <IconButton
                  size="small"
                  color="error"
                  onClick={(e) => handleDelete(task.id, task.name, e)}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          </Box>
        </Paper>
      ))}
    </Stack>
  );
}
