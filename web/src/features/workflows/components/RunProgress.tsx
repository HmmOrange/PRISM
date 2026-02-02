import { Box, CircularProgress, Typography } from "@mui/material";
import { useEffect, useRef } from "react";

import {
  startWorkflowGeneration,
  getWorkflowGenerationStatus,
} from "../../../api/workflows.api";
import type { RunWorkflowWizardState } from "../hooks/useRunWorkflowWizard";

interface Props {
  state: RunWorkflowWizardState;
  setJob: (jobId: string, status: RunWorkflowWizardState["jobStatus"]) => void;
  setJobStatus: (status: RunWorkflowWizardState["jobStatus"]) => void;
  nextStep: () => void;
}

export default function RunProgress({
  state,
  setJob,
  setJobStatus,
  nextStep,
}: Props) {
  const startedRef = useRef(false);

  // Start job ONCE
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    startWorkflowGeneration({
      task_ids: state.selectedTaskIds,
      pipeline_path: state.pipelinePath!,
      rounds: state.rounds,
      run_after: state.runMode !== "GENERATE_ONLY",
      calculate_after:
        state.runMode === "GENERATE_RUN_AND_SCORE",
    }).then((res) => {
      setJob(res.job_id, "RUNNING");
    });
  }, []);

  // Poll job status
  useEffect(() => {
    if (!state.jobId || state.jobStatus !== "RUNNING") return;

    const interval = setInterval(() => {
      getWorkflowGenerationStatus(state.jobId!)
        .then((res) => {
          if (res.status === "DONE") {
            setJobStatus("DONE");
            clearInterval(interval);
            nextStep();
          }

          if (res.status === "FAILED") {
            setJobStatus("FAILED");
            clearInterval(interval);
          }
        })
        .catch(() => {
          setJobStatus("FAILED");
          clearInterval(interval);
        });
    }, 1000);

    return () => clearInterval(interval);
  }, [state.jobId, state.jobStatus]);

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      gap={3}
      mt={6}
    >
      <CircularProgress />
      <Typography variant="h6">Running workflows…</Typography>
      <Typography variant="body2" color="text.secondary">
        This may take several minutes depending on task size.
      </Typography>
    </Box>
  );
}
