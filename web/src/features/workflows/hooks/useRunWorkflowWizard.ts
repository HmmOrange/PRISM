import { useState } from "react";

export type RunMode =
  | "GENERATE_ONLY"
  | "GENERATE_AND_RUN"
  | "GENERATE_RUN_AND_SCORE";

export interface RunWorkflowWizardState {
  // Step control
  activeStep: number;

  // Step 1: task selection
  selectedTaskIds: string[];

  // Step 2: configuration
  pipelinePath: string | null;
  runMode: RunMode;
  rounds: number;

  // Step 3: execution
  jobId: string | null;
  jobStatus: "IDLE" | "RUNNING" | "DONE" | "FAILED";
}

export function useRunWorkflowWizard() {
  const [state, setState] = useState<RunWorkflowWizardState>({
    activeStep: 0,

    selectedTaskIds: [],

    pipelinePath: null,
    runMode: "GENERATE_ONLY",
    rounds: 1,

    jobId: null,
    jobStatus: "IDLE",
  });

  // --------------------
  // Step navigation
  // --------------------

  function nextStep() {
    setState((prev) => ({
      ...prev,
      activeStep: Math.min(prev.activeStep + 1, 3),
    }));
  }

  function prevStep() {
    setState((prev) => ({
      ...prev,
      activeStep: Math.max(prev.activeStep - 1, 0),
    }));
  }

  function goToStep(step: number) {
    setState((prev) => ({
      ...prev,
      activeStep: step,
    }));
  }

  // --------------------
  // Step 1 setters
  // --------------------

  function setSelectedTaskIds(taskIds: string[]) {
    setState((prev) => ({
      ...prev,
      selectedTaskIds: taskIds,
    }));
  }

  // --------------------
  // Step 2 setters
  // --------------------

  function setPipelinePath(path: string | null) {
    setState((prev) => ({
      ...prev,
      pipelinePath: path,
    }));
  }

  function setRunMode(mode: RunMode) {
    setState((prev) => ({
      ...prev,
      runMode: mode,
    }));
  }

  function setRounds(rounds: number) {
    setState((prev) => ({
      ...prev,
      rounds,
    }));
  }

  // --------------------
  // Step 3 setters
  // --------------------

  function setJob(jobId: string, status: RunWorkflowWizardState["jobStatus"]) {
    setState((prev) => ({
      ...prev,
      jobId,
      jobStatus: status,
    }));
  }

  function setJobStatus(status: RunWorkflowWizardState["jobStatus"]) {
    setState((prev) => ({
      ...prev,
      jobStatus: status,
    }));
  }

  // --------------------
  // Reset (optional but useful)
  // --------------------

  function resetWizard() {
    setState({
      activeStep: 0,

      selectedTaskIds: [],

      pipelinePath: null,
      runMode: "GENERATE_ONLY",
      rounds: 1,

      jobId: null,
      jobStatus: "IDLE",
    });
  }

  return {
    state,

    // navigation
    nextStep,
    prevStep,
    goToStep,

    // step 1
    setSelectedTaskIds,

    // step 2
    setPipelinePath,
    setRunMode,
    setRounds,

    // step 3
    setJob,
    setJobStatus,

    // misc
    resetWizard,
  };
}
