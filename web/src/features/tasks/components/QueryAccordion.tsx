import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  TextField,
  Select,
  MenuItem,
  Divider,
  Box,
  FormControl,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useState } from "react";
import QueryHeader from "./QueryHeader";
import QueryFilesPanel from "./QueryFilesPanel";
import ConfirmDialog from "./ConfirmDialog";
import type { QueryData, DatasetSplit } from "../types";

interface Props {
  query: QueryData;
  onUpdate: (q: QueryData) => void;
  onDeleteFile: (fileId: string) => void;
  onRemoveQuery: () => void;
}

export default function QueryAccordion({
  query,
  onUpdate,
  onDeleteFile,
  onRemoveQuery,
}: Props) {
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <>
      <Accordion>
        <AccordionSummary
          component="div"
          expandIcon={<ExpandMoreIcon />}
          sx={{ cursor: "pointer" }}
        >
          <QueryHeader
            query={query}
            onRemove={() => setConfirmOpen(true)}
          />
        </AccordionSummary>

        <AccordionDetails>
          <Stack spacing={2}>
            <Divider />

            {/* Row 1: Query name + split */}
            <Box display="flex" gap={2}>
              <TextField
                fullWidth
                size="small"
                label="Query name"
                value={query.name}
                onChange={(e) =>
                  onUpdate({ ...query, name: e.target.value })
                }
                sx={{ flex: 3 }}
              />

              <FormControl size="small" sx={{ flex: 1 }}>
                <Select
                  value={query.split}
                  onChange={(e) =>
                    onUpdate({
                      ...query,
                      split: e.target.value as DatasetSplit,
                    })
                  }
                >
                  <MenuItem value="test">Test</MenuItem>
                  <MenuItem value="validation">Validation</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {/* Row 2: Label */}
            <TextField
              fullWidth
              size="small"
              label="Label"
              multiline
              minRows={1}
              value={query.label}
              onChange={(e) =>
                onUpdate({ ...query, label: e.target.value })
              }
            />

            {/* Row 3: Files */}
            <QueryFilesPanel
              query={query}
              onUpdate={onUpdate}
              onDeleteFile={onDeleteFile}
            />
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* Confirmation dialog */}
      <ConfirmDialog
        open={confirmOpen}
        title="Delete query?"
        description={`This will permanently remove "${
          query.name || `Query ${query.id}`
        }" and all its files.`}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => {
          setConfirmOpen(false);
          onRemoveQuery();
        }}
      />
    </>
  );
}
