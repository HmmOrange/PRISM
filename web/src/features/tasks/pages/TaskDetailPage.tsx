import {
  Container,
  Typography,
  Stack,
  CircularProgress,
  Box,
} from "@mui/material";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import { getTask } from "../../../api/tasks.api";
import type { TaskDetail } from "../../../types/tasks.types";
import SectionCard from "../components/SectionCard";
import QueryAccordion from "../components/QueryAccordion";

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<TaskDetail | null>(null);

  useEffect(() => {
    if (!taskId) return;
    getTask(taskId).then(setTask);
  }, [taskId]);

  useEffect(() => {
    if (task) {
      console.log("TASK DETAIL:", task);
    }
  }, [task]);


  if (!task) {
    return (
      <Container sx={{ mt: 6, display: "flex", justifyContent: "center" }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <Stack spacing={4}>
        {/* ===== Task Metadata ===== */}
        <SectionCard title="Task Metadata">
          <Stack spacing={2}>
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Name
              </Typography>
              <Typography>{task.name}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Metric
              </Typography>
              <Typography>{task.metric}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Description
              </Typography>
              <Typography whiteSpace="pre-line">
                {task.description || "—"}
              </Typography>
            </Box>
          </Stack>
        </SectionCard>

        {/* ===== Dataset ===== */}
        <SectionCard title="Dataset">
          <Stack spacing={2}>
            {task.queries.map((q) => (
              <QueryAccordion
                key={q.index}
                query={q}
                mode="view"
              />
            ))}
          </Stack>
        </SectionCard>
      </Stack>
    </Container>
  );
}
