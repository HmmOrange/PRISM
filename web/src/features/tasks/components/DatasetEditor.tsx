import { Stack, Button, Box, IconButton, Tooltip, Typography } from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import QueryAccordion from "./QueryAccordion";
import type { EditableQuery } from "../../../types/tasks.types";

interface Props {
  queries: EditableQuery[];
  setQueries: (q: EditableQuery[]) => void;
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

  function updateQuery(updated: EditableQuery) {
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
      {queries.length === 0 ? (
        <Box sx={{ py: 4, textAlign: "center" }}>
          <Typography color="text.secondary">
            No queries yet. Click "Add Query" to get started.
          </Typography>
        </Box>
      ) : (
        queries.map((q, index) => (
          <Box
            key={q.id}
            sx={{
              position: "relative",
              "&:hover .query-number": { opacity: 1 },
              "&:hover .delete-query-btn": { opacity: 1 },
            }}
          >
            {/* Query number indicator */}
            <Typography
              className="query-number"
              variant="caption"
              sx={{
                position: "absolute",
                left: -24,
                top: "50%",
                transform: "translateY(-50%)",
                opacity: 0,
                transition: "opacity 0.2s",
                color: "text.secondary",
                fontWeight: 500,
              }}
            >
              {index + 1}
            </Typography>

            <QueryAccordion
              query={q}
              onUpdate={updateQuery}
              onDeleteFile={(fid) => deleteFile(q.id, fid)}
              mode="edit"
            />

            {/* Delete button */}
            <Tooltip title="Remove Query" placement="right">
              <IconButton
                className="delete-query-btn"
                size="small"
                onClick={() => removeQuery(q.id)}
                sx={{
                  position: "absolute",
                  right: -40,
                  top: "50%",
                  transform: "translateY(-50%)",
                  opacity: 0,
                  transition: "opacity 0.2s",
                  color: "error.main",
                  "&:hover": {
                    backgroundColor: "error.light",
                    color: "error.contrastText",
                  },
                }}
              >
                <DeleteOutlineIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        ))
      )}

      <Button variant="outlined" onClick={addQuery} sx={{ alignSelf: "flex-start" }}>
        + Add Query
      </Button>
    </Stack>
  );
}
