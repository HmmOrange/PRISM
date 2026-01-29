
import Grid from "@mui/material/Grid";
import TaskCard from "./TaskCard";
import type { TaskListItem } from "../../../types/tasks.types";

interface Props {
  tasks: TaskListItem[];
}

export default function TaskCardGrid({ tasks }: Props) {
  return (
    <Grid
      container
      spacing={3}
      columns={{ xs: 1, sm: 12, md: 12 }}
    >
      {tasks.map((task) => (
        <Grid
          key={task.id}
          size={{ xs: 12, sm: 6, md: 4 }}
        >
          <TaskCard task={task} />
        </Grid>
      ))}
    </Grid>
  );
}
