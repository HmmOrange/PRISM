import { Stack, Chip } from "@mui/material";

interface Props {
  test: number;
  validation: number;
}

export default function TaskStats({ test, validation }: Props) {
  return (
    <Stack direction="row" spacing={1}>
      <Chip label={`Test ${test}`} size="small" color="primary" />
      <Chip
        label={`Validation ${validation}`}
        size="small"
        color="secondary"
      />
    </Stack>
  );
}
