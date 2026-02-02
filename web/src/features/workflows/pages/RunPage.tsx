import { Container, Box, Button } from "@mui/material";

import StepperHeader from "../components/StepperHeader";
import TaskPicker from "../components/TaskPicker";
import PipelinePicker from "../components/PipelinePicker";
import RunProgress from "../components/RunProgress";
import RunResults from "../components/RunResults";

import { useRunWorkflowWizard } from "../hooks/useRunWorkflowWizard";

export default function RunPage() {
  const {
    state,

    nextStep,
    prevStep,

    setSelectedTaskIds,
    setPipelinePath,
    setRunMode,
    setRounds,

    setJob,
    setJobStatus,
  } = useRunWorkflowWizard();

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <StepperHeader activeStep={state.activeStep} />

      <Box mt={4}>
        {state.activeStep === 0 && (
          <TaskPicker
            selectedTaskIds={state.selectedTaskIds}
            onChange={setSelectedTaskIds}
          />
        )}

        {state.activeStep === 1 && (
          <PipelinePicker
            pipelinePath={state.pipelinePath}
            runMode={state.runMode}
            rounds={state.rounds}
            onPipelineChange={setPipelinePath}
            onRunModeChange={setRunMode}
            onRoundsChange={setRounds}
          />
        )}

        {state.activeStep === 2 && (
          <RunProgress
            state={state}
            setJob={setJob}
            setJobStatus={setJobStatus}
            nextStep={nextStep}
          />
        )}

        {state.activeStep === 3 && <RunResults state={state}/>}
      </Box>

      <Box
        mt={4}
        display="flex"
        justifyContent="space-between"
      >
        <Button
          disabled={state.activeStep === 0}
          onClick={prevStep}
        >
          Back
        </Button>

        <Button
          variant="contained"
          onClick={nextStep}
          disabled={
            (state.activeStep === 0 &&
              state.selectedTaskIds.length === 0) ||
            (state.activeStep === 1 &&
              !state.pipelinePath)
          }
        >
          {state.activeStep < 3 ? "Next" : "Finish"}
        </Button>
      </Box>
    </Container>
  );
}
