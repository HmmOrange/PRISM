import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Typography,
  Chip,
  Box,
  Tooltip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import QueryViewPanel from "./QueryViewPanel.tsx";
import QueryEditPanel from "./QueryEditPanel.tsx";

import type {
  EditableQuery,
  QueryDetail,
} from "../../../types/tasks.types";

interface Props {
  query: EditableQuery | QueryDetail;
  mode: "view" | "edit";
  onUpdate?: (q: EditableQuery) => void;
  onDeleteFile?: (fileId: string) => void;
}

export default function QueryAccordion({
  query,
  mode,
  onUpdate,
  onDeleteFile,
}: Props) {
  const queryIndex =
    "index" in query ? query.index : query.id;

  const displayName = mode === "edit"
                ? (query as EditableQuery).name || `Query ${queryIndex + 1}`
                : ("name" in query && query.name) || `Query ${queryIndex + 1}`;

  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Stack
          direction="row"
          alignItems="center"
          spacing={2}
          sx={{ width: "100%" }}
        >
          <Typography fontWeight={600}>
            {displayName}
          </Typography>

          <Chip
            size="small"
            label={query.split}
            color={query.split === "test" ? "primary" : "secondary"}
          />

          {query.label && (
            <Tooltip title="Ground truth label">
              <Chip
                size="small"
                label={query.label}
                variant="outlined"
                color="success"
              />
            </Tooltip>
          )}

          <Box sx={{ flexGrow: 1 }} />

          <Typography
            variant="caption"
            color="text.secondary"
          >
            {query.files.length} file{query.files.length !== 1 ? "s" : ""}
          </Typography>
        </Stack>
      </AccordionSummary>

      <AccordionDetails>
        {mode === "view" ? (
          <QueryViewPanel query={query as QueryDetail} />
        ) : (
          <QueryEditPanel
            query={query as EditableQuery}
            onUpdate={onUpdate!}
            onDeleteFile={onDeleteFile!}
          />
        )}
      </AccordionDetails>
    </Accordion>
  );
}
