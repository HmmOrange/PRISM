import { Stack, TextField } from "@mui/material";
import QueryFilesPanel from "./QueryFilesPanel";

import type { EditableQuery } from "../../../types/tasks.types";

interface Props {
  query: EditableQuery;
  onUpdate: (q: EditableQuery) => void;
  onDeleteFile: (fileId: string) => void;
}

export default function QueryEditPanel({
  query,
  onUpdate,
  onDeleteFile,
}: Props) {
  return (
    <Stack spacing={2}>
      <TextField
        label="Query name"
        size="small"
        value={query.name}
        onChange={(e) =>
          onUpdate({ ...query, name: e.target.value })
        }
      />

      <TextField
        label="Label"
        size="small"
        multiline
        minRows={1}
        value={query.label}
        onChange={(e) =>
          onUpdate({ ...query, label: e.target.value })
        }
      />

      <QueryFilesPanel
        query={query}
        onUpdate={onUpdate}
        onDeleteFile={onDeleteFile}
      />
    </Stack>
  );
}
