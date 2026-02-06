import { Button, Box, Typography, Paper, Stack } from "@mui/material";
import TaskMetaForm from "./TaskMetaForm";
import DatasetEditor from "./DatasetEditor";
import type { EditableQuery } from "../../../types/tasks.types";

interface TaskMeta {
  name: string;
  metric: string;
  description: string;
}

interface ValidationErrors {
  name?: string;
  metric?: string;
}

interface Props {
  meta: TaskMeta;
  queries: EditableQuery[];
  onMetaChange: (field: string, value: string) => void;
  onQueriesChange: (queries: EditableQuery[]) => void;
  onSubmit: () => void;
  submitting?: boolean;
  errors?: ValidationErrors;
}

export default function CreateTaskTab({
  meta,
  queries,
  onMetaChange,
  onQueriesChange,
  onSubmit,
  submitting = false,
  errors = {},
}: Props) {
  return (
    <Box>
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={3}
        alignItems="stretch"
      >
        {/* Left side - Task Metadata */}
        <Box sx={{ flex: { xs: "1", md: "0 0 40%" }, minWidth: 0 }}>
          <Paper variant="outlined" sx={{ p: 3, height: "100%" }}>
            <Typography variant="h6" gutterBottom>
              Task Information
            </Typography>
            <TaskMetaForm
              name={meta.name}
              metric={meta.metric}
              description={meta.description}
              onChange={onMetaChange}
              errors={errors}
            />
          </Paper>
        </Box>

        {/* Right side - Dataset Editor */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Paper variant="outlined" sx={{ p: 3, minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Dataset
            </Typography>
            <DatasetEditor
              queries={queries}
              setQueries={onQueriesChange}
            />
          </Paper>
        </Box>
      </Stack>

      {/* Submit Button - Full width */}
      <Box display="flex" justifyContent="flex-end" pt={3}>
        <Button
          variant="contained"
          size="large"
          onClick={onSubmit}
          disabled={submitting}
        >
          {submitting ? "Creating..." : "Create Task"}
        </Button>
      </Box>
    </Box>
  );
}
