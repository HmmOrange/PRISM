import { useEffect, useState } from "react";
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  Button,
  Collapse,
} from "@mui/material";

import {
  getWorkflowResults,
  type WorkflowResult,
} from "../../../api/workflows.api";
import type { RunWorkflowWizardState } from "../hooks/useRunWorkflowWizard";
import { API_CONFIG } from "../../../config/api";

interface Props {
  state: RunWorkflowWizardState;
}

export default function RunResults({ state }: Props) {
  const [results, setResults] = useState<WorkflowResult[] | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [contents, setContents] = useState<Record<string, string>>({});
  const [loadingContent, setLoadingContent] = useState<Record<string, boolean>>(
    {}
  );

  useEffect(() => {
    if (!state.jobId) return;
    getWorkflowResults(state.jobId).then(setResults);
  }, [state.jobId]);

  async function loadContent(r: WorkflowResult) {
    const key = `${r.task}/${r.filename}`;
    if (contents[key]) return;

    setLoadingContent((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await fetch(`${API_CONFIG.baseUrl}${r.download_url}`);
      const text = await res.text();
      setContents((prev) => ({ ...prev, [key]: text }));
    } finally {
      setLoadingContent((prev) => ({ ...prev, [key]: false }));
    }
  }

  function toggle(r: WorkflowResult) {
    const key = `${r.task}/${r.filename}`;
    setExpanded((prev) => {
      const next = !prev[key];
      if (next) loadContent(r);
      return { ...prev, [key]: next };
    });
  }

  if (!results) {
    return <CircularProgress />;
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Generated Workflows
      </Typography>

      {results.length === 0 && (
        <Typography>No workflows generated.</Typography>
      )}

      {results.map((r) => {
        const key = `${r.task}/${r.filename}`;
        const isOpen = !!expanded[key];

        return (
          <Paper
            key={key}
            variant="outlined"
            sx={{ p: 2, mb: 2 }}
          >
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
            >
              <Typography fontWeight={500}>
                {r.task}
              </Typography>

              <Box display="flex" gap={1}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => toggle(r)}
                >
                  {isOpen ? "Hide" : "View"}
                </Button>

                <Button
                  size="small"
                  variant="contained"
                  component="a"
                  href={r.download_url}
                  target="_blank"
                >
                  Download
                </Button>
              </Box>
            </Box>

            <Collapse in={isOpen}>
              <Box mt={2}>
                {loadingContent[key] ? (
                  <CircularProgress size={20} />
                ) : (
                  <Box
                    component="pre"
                    sx={{
                      backgroundColor: "#0f172a",
                      color: "#e5e7eb",
                      p: 2,
                      borderRadius: 1,
                      fontSize: 13,
                      lineHeight: 1.6,
                      overflowX: "auto",
                      maxHeight: 500,
                    }}
                  >
                    {contents[key]}
                  </Box>
                )}
              </Box>
            </Collapse>
          </Paper>
        );
      })}
    </Box>
  );
}
