import { Card, CardContent, Typography, Stack, Chip } from "@mui/material";
import TaskStats from "./TaskStats";
import type { TaskListItem } from "../../../types/tasks.types";

interface Props {
  task: TaskListItem;
}

export default function TaskCard({ task }: Props) {
  return (
    <Card
      sx={{
        height: "100%",
        transition: "0.2s",
        "&:hover": { boxShadow: 4 },
      }}
    >
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

          <Chip label={task.metric} size="small" sx={{ width: "fit-content" }} />

          <TaskStats
            test={task.test_queries}
            validation={task.validation_queries}
          />
        </Stack>
      </CardContent>
    </Card>
  );
}
