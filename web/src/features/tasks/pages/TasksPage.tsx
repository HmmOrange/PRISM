import {
  Container,
  Typography,
  CircularProgress,
  Box,
} from "@mui/material";
import { useEffect, useState } from "react";

import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import TaskCardGrid from "../components/TaskCardGrid";

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    reloadTasks();
  }, []);

  function reloadTasks() {
    setLoading(true);
    getTasks()
      .then(setTasks)
      .finally(() => setLoading(false));
  }

  function handleTaskDeleted(taskId: string) {
    setTasks((prev) => prev.filter((t) => t.id !== taskId));
  }

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" gutterBottom>
        Tasks
      </Typography>

      {loading ? (
        <Box display="flex" justifyContent="center" mt={6}>
          <CircularProgress />
        </Box>
      ) : (
        <TaskCardGrid
          tasks={tasks}
          onTaskDeleted={handleTaskDeleted}
        />
      )}
    </Container>
  );
}
