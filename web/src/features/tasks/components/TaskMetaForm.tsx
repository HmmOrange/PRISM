import { Stack, TextField } from "@mui/material";

interface Props {
  name: string;
  metric: string;
  description: string;
  onChange: (field: string, value: string) => void;
}

export default function TaskMetaForm({
  name,
  metric,
  description,
  onChange,
}: Props) {
  return (
    <Stack spacing={2}>
      <TextField
        label="Task Name"
        value={name}
        onChange={(e) => onChange("name", e.target.value)}
      />

      <TextField
        label="Metric"
        placeholder="accuracy, f1, bleu, ..."
        value={metric}
        onChange={(e) => onChange("metric", e.target.value)}
      />

      <TextField
        label="Task Description"
        multiline
        rows={4}
        value={description}
        onChange={(e) => onChange("description", e.target.value)}
      />
    </Stack>
  );
}
