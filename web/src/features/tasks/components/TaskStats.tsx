import { Stack } from "@mui/material";

import { Tag } from "../../../components";

interface Props {
  test: number;
  validation: number;
}

export default function TaskStats({ test, validation }: Props) {
  return (
    <Stack direction="row" spacing={1}>
      <Tag label={`Test ${test}`} size="small" variant="test" />
      <Tag label={`Validation ${validation}`} size="small" variant="validation" />
    </Stack>
  );
}
