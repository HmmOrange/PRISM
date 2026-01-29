import {
  Container,
  Typography,
  Stack,
  Chip,
  CircularProgress,
} from "@mui/material";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import { getTask } from "../../../api/tasks.api";
import type { TaskDetail } from "../../../types/tasks.types";

export default function TaskDetailPage() {
  const { taskId } = useParams();
  const [task, setTask] = useState<TaskDetail | null>(null);

  useEffect(() => {
    if (!taskId) return;
    getTask(taskId).then(setTask);
  }, [taskId]);

  if (!task) {
    return (
      <Container sx={{ mt: 6, display: "flex", justifyContent: "center" }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <Stack spacing={3}>
        <Typography variant="h4">{task.name}</Typography>

        <Typography color="text.secondary">
          {task.description}
        </Typography>

        <Chip label={task.metric} sx={{ width: "fit-content" }} />

        <Stack spacing={1}>
          <Typography variant="h6">Queries</Typography>

          {task.queries.map((q) => (
            <Typography key={q.index}>
              #{q.index} — {q.split}
              {q.label && ` — ${q.label}`}
            </Typography>
          ))}
        </Stack>
      </Stack>
    </Container>
  );
}
