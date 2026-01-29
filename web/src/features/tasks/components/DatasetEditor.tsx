import { Stack, Button } from "@mui/material";
import QueryAccordion from "./QueryAccordion";
import type { QueryData } from "../types";

interface Props {
  queries: QueryData[];
  setQueries: (q: QueryData[]) => void;
}

export default function DatasetEditor({ queries, setQueries }: Props) {
  function addQuery() {
    setQueries([
      ...queries,
      {
        id: queries.length,
        name: "",
        split: "test",
        label: "",
        files: [],
      },
    ]);
  }

  function updateQuery(updated: QueryData) {
    setQueries(
      queries.map((q) => (q.id === updated.id ? updated : q))
    );
  }

  function removeQuery(queryId: number) {
    const filtered = queries.filter((q) => q.id !== queryId);
    // reindex to keep 0..N-1
    setQueries(filtered.map((q, idx) => ({ ...q, id: idx })));
  }

  function deleteFile(queryId: number, fileId: string) {
    setQueries(
      queries.map((q) =>
        q.id === queryId
          ? { ...q, files: q.files.filter((f) => f.id !== fileId) }
          : q
      )
    );
  }

  return (
    <Stack spacing={2}>
      {queries.map((q) => (
        <QueryAccordion
          key={q.id}
          query={q}
          onUpdate={updateQuery}
          onDeleteFile={(fid) => deleteFile(q.id, fid)}
          onRemoveQuery={() => removeQuery(q.id)}
        />
      ))}

      <Button onClick={addQuery}>Add Query</Button>
    </Stack>
  );
}
