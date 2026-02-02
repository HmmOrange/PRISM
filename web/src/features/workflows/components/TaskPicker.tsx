import {
  Box,
  Checkbox,
  CircularProgress,
  Container,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";

import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";

interface Props {
  selectedTaskIds: string[];
  onChange: (taskIds: string[]) => void;
}

export default function TaskPicker({
  selectedTaskIds,
  onChange,
}: Props) {
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getTasks()
      .then(setTasks)
      .finally(() => setLoading(false));
  }, []);

  function toggleTask(taskId: string) {
    if (selectedTaskIds.includes(taskId)) {
      onChange(selectedTaskIds.filter((id) => id !== taskId));
    } else {
      onChange([...selectedTaskIds, taskId]);
    }
  }

  return (
    <Container sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        Select Tasks
      </Typography>

      {loading ? (
        <Box display="flex" justifyContent="center" mt={4}>
          <CircularProgress />
        </Box>
      ) : (
        <Box display="flex" flexDirection="column" gap={1}>
          {tasks.map((task) => (
            <Box
              key={task.id}
              display="flex"
              alignItems="center"
              gap={1}
              sx={{
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 1,
                px: 2,
                py: 1,
              }}
            >
              <Checkbox
                checked={selectedTaskIds.includes(task.id)}
                onChange={() => toggleTask(task.id)}
              />

              <Box>
                <Typography variant="subtitle1">
                  {task.name}
                </Typography>

                {task.description && (
                  <Typography
                    variant="body2"
                    color="text.secondary"
                  >
                    {task.description}
                  </Typography>
                )}
              </Box>
            </Box>
          ))}
        </Box>
      )}
    </Container>
  );
}
