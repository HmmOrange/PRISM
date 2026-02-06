/**
 * Task List View component.
 * Alternative view for task library.
 */

import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Typography,
  Box,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import DeleteIcon from "@mui/icons-material/Delete";
import VisibilityIcon from "@mui/icons-material/Visibility";

import type { TaskListItem } from "../../../types/tasks.types";
import { getMetricLabel } from "../../../config/metrics";
import { deleteTask } from "../../../api/tasks.api";
import { useToast } from "../../../components/feedback/ToastProvider";

interface TaskListViewProps {
  tasks: TaskListItem[];
  onTaskDeleted: (taskId: string) => void;
}

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

  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Description</TableCell>
            <TableCell>Metric</TableCell>
            <TableCell align="center">Queries</TableCell>
            <TableCell>Created</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {tasks.map((task) => (
            <TableRow
              key={task.id}
              hover
              sx={{ cursor: "pointer" }}
              onClick={() => navigate(`/tasks/${task.id}`)}
            >
              <TableCell>
                <Typography fontWeight={500}>{task.name}</Typography>
              </TableCell>
              <TableCell>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  noWrap
                  sx={{ maxWidth: 200 }}
                >
                  {task.description || "—"}
                </Typography>
              </TableCell>
              <TableCell>
                <Chip
                  label={getMetricLabel(task.metric)}
                  size="small"
                  variant="outlined"
                />
              </TableCell>
              <TableCell align="center">
                <Box display="flex" gap={1} justifyContent="center">
                  <Tooltip title="Validation queries">
                    <Chip
                      size="small"
                      label={`V: ${task.validation_queries}`}
                      color="success"
                      variant="outlined"
                    />
                  </Tooltip>
                  <Tooltip title="Test queries">
                    <Chip
                      size="small"
                      label={`T: ${task.test_queries}`}
                      color="info"
                      variant="outlined"
                    />
                  </Tooltip>
                </Box>
              </TableCell>
              <TableCell>
                <Typography variant="body2" color="text.secondary">
                  {formatDate(task.created_at)}
                </Typography>
              </TableCell>
              <TableCell align="right">
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
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
