import {
  Container,
  Typography,
  Stack,
  Chip,
  CircularProgress,
  Divider,
  Box,
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
            <Typography color="text.secondary">{task.description}</Typography>
            <Chip label={task.metric} sx={{ width: "fit-content" }} />

            <Divider />

            <Typography variant="h6">Queries</Typography>

            <Stack spacing={2}>
            {task.queries.map((q) => (
                <Box
                key={q.index}
                sx={{
                    p: 2,
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 2,
                }}
                >
                <Stack spacing={1}>
                    <Typography fontWeight={600}>
                    Query {q.index} · {q.split}
                    </Typography>

                    {q.label && (
                    <Typography color="text.secondary">
                        Label: {q.label}
                    </Typography>
                    )}

                    {q.files.length === 0 ? (
                    <Typography color="text.secondary">
                        No files uploaded
                    </Typography>
                    ) : (
                    <Stack spacing={0.5}>
                        {q.files.map((f) => (
                        <Typography
                            key={f.object_key}
                            sx={{ fontFamily: "monospace", fontSize: 13 }}
                        >
                            {f.filename} · {(f.size / 1024).toFixed(1)} KB
                        </Typography>
                        ))}
                    </Stack>
                    )}
                </Stack>
                </Box>
            ))}
            </Stack>
        </Stack>
        </Container>
  );
}
