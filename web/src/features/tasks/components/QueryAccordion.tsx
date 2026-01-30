import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Typography,
  Chip,
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

  const title = mode === "edit"
                ? (query as EditableQuery).name || `Query ${queryIndex + 1}`
                : `Query ${queryIndex + 1}`;

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
            {title}
          </Typography>

          <Chip
            size="small"
            label={query.split}
            color={query.split === "test" ? "primary" : "secondary"}
          />

          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ ml: "auto" }}
          >
            {query.files.length} files
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
