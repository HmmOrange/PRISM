import { Stack, Typography } from "@mui/material";
import QueryFilesPanel from "./QueryFilesPanel";
import type { QueryDetail } from "../../../types/tasks.types";

interface Props {
  query: QueryDetail;
}

export default function QueryViewPanel({ query }: Props) {
  return (
    <Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">
        Label
      </Typography>

      <Typography>
        {query.label || "—"}
      </Typography>

      <QueryFilesPanel query={query} readOnly />
    </Stack>
  );
}
