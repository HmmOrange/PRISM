import {
  Stack,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
} from "@mui/material";
import { AVAILABLE_METRICS } from "../../../config/metrics";

interface ValidationErrors {
  name?: string;
  metric?: string;
}

interface Props {
  name: string;
  metric: string;
  description: string;
  onChange: (field: string, value: string) => void;
  errors?: ValidationErrors;
}

export default function TaskMetaForm({
  name,
  metric,
  description,
  onChange,
  errors = {},
}: Props) {
  return (
    <Stack spacing={2.5}>
      <TextField
        label="Task Name"
        value={name}
        onChange={(e) => onChange("name", e.target.value)}
        fullWidth
        required
        error={!!errors.name}
        helperText={errors.name}
      />

      <FormControl fullWidth required error={!!errors.metric}>
        <InputLabel id="metric-select-label">Metric</InputLabel>
        <Select
          labelId="metric-select-label"
          value={metric}
          label="Metric"
          onChange={(e) => onChange("metric", e.target.value)}
        >
          {AVAILABLE_METRICS.map((m) => (
            <MenuItem key={m.value} value={m.value}>
              {m.label}
            </MenuItem>
          ))}
        </Select>
        {errors.metric && <FormHelperText>{errors.metric}</FormHelperText>}
      </FormControl>

      <TextField
        label="Task Description"
        multiline
        rows={4}
        value={description}
        onChange={(e) => onChange("description", e.target.value)}
        fullWidth
      />
    </Stack>
  );
}
