/**
 * Create Task Wizard Page.
 * Implements SRS 2.3.1 - Multi-Step Task Creation Wizard
 * 
 * Stages:
 * 1. Metadata Configuration
 * 2. Pipeline Configuration
 * 3. Dataset Ingestion
 * 4. Review & Finalization
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Container, Paper, Box, Typography, Fade } from "@mui/material";

import {
  WizardStepper,
  WizardNavigation,
  MetadataStep,
  PipelineStep,
  DatasetStep,
  ReviewStep,
  TaskSuccessDialog,
} from "../components/wizard";

import { useTaskWizard } from "../hooks/useTaskWizard";
import { createTask, commitTaskFiles } from "../../../api/tasks.api";
import { uploadTaskFiles } from "../../../utils/uploadExecutor";
import { useToast } from "../../../components/feedback/ToastProvider";
import { ROUTES } from "../../../config/routes";

const WIZARD_STEPS = [
  "Metadata",
  "Pipeline",
  "Dataset",
  "Review",
];

export default function CreateTaskWizardPage() {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [successDialog, setSuccessDialog] = useState<{
    open: boolean;
    taskId: string;
    taskName: string;
  }>({ open: false, taskId: "", taskName: "" });

  const {
    state,
    errors,
    totalSteps,
    nextStep,
    prevStep,
    goToStep,
    canProceed,
    setMetadata,
    togglePipelineTag,
    setQueries,
    setSubmitting,
  } = useTaskWizard();

  /**
   * Handle task creation submission.
   */
  async function handleSubmit() {
    if (state.isSubmitting) return;

    setSubmitting(true);
    try {
      const payload = {
        name: state.name,
        metric: state.metrics[0] || "", // Primary metric (API expects single for now)
        description: state.description,
        queries: state.queries.map((q) => ({
          id: q.id,
          name: q.name || "",
          split: q.split,
          label: q.label || "",
          files: q.files.map((f) => ({
            filename: f.file.name,
            content_type: f.file.type,
          })),
        })),
      };

      const result = await createTask(payload);

      // Upload files if any
      const hasFiles = state.queries.some((q) => q.files.length > 0);
      if (hasFiles) {
        const committedFiles = await uploadTaskFiles(result, state.queries);
        await commitTaskFiles(result.task_id, committedFiles);
      }

      // Show success dialog
      setSuccessDialog({
        open: true,
        taskId: result.task_id,
        taskName: state.name,
      });

      showToast({ message: `Task "${state.name}" created successfully`, severity: "success" });
    } catch (err) {
      console.error("Task creation failed", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to create task";
      setSubmitting(false, errorMessage);
      showToast({ message: errorMessage, severity: "error" });
    }
  }

  /**
   * Handle viewing the created task.
   */
  function handleViewTask() {
    navigate(`/tasks/${successDialog.taskId}`);
  }

  /**
   * Handle generating workflow (disabled for Phase 1).
   */
  function handleGenerateWorkflow() {
    // Navigate to run page with task pre-selected
    navigate(ROUTES.public.runTasks, {
      state: { selectedTaskId: successDialog.taskId },
    });
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        Create New Task
      </Typography>
      <Typography color="text.secondary" mb={3}>
        Follow the steps below to configure and create your ML task.
      </Typography>

      <Paper variant="outlined" sx={{ p: { xs: 2, md: 4 } }}>
        {/* Wizard Stepper */}
        <WizardStepper
          activeStep={state.activeStep}
          steps={WIZARD_STEPS}
          onStepClick={goToStep}
        />

        {/* Step Content with Animation */}
        <Box mt={4}>
          <Fade in key={state.activeStep} timeout={300}>
            <Box>
              {/* Step 1: Metadata */}
              {state.activeStep === 0 && (
                <MetadataStep
                  name={state.name}
                  description={state.description}
                  metrics={state.metrics}
                  errors={errors}
                  onNameChange={(value) => setMetadata({ name: value })}
                  onDescriptionChange={(value) => setMetadata({ description: value })}
                  onMetricsChange={(metrics) => setMetadata({ metrics })}
                />
              )}

              {/* Step 2: Pipeline Configuration */}
              {state.activeStep === 1 && (
                <PipelineStep
                  selectedTags={state.pipelineTags}
                  onToggleTag={togglePipelineTag}
                />
              )}

              {/* Step 3: Dataset */}
              {state.activeStep === 2 && (
                <DatasetStep
                  queries={state.queries}
                  onQueriesChange={setQueries}
                />
              )}

              {/* Step 4: Review */}
              {state.activeStep === 3 && (
                <ReviewStep
                  name={state.name}
                  description={state.description}
                  metrics={state.metrics}
                  pipelineTags={state.pipelineTags}
                  queries={state.queries}
                  onEditStep={goToStep}
                  submitError={state.submitError}
                />
              )}
            </Box>
          </Fade>
        </Box>

        {/* Navigation Buttons */}
        <WizardNavigation
          activeStep={state.activeStep}
          totalSteps={totalSteps}
          canProceed={canProceed()}
          isSubmitting={state.isSubmitting}
          onBack={prevStep}
          onNext={nextStep}
          onSubmit={handleSubmit}
          submitLabel="Create Task"
        />
      </Paper>

      {/* Success Dialog */}
      <TaskSuccessDialog
        open={successDialog.open}
        taskId={successDialog.taskId}
        taskName={successDialog.taskName}
        onViewTask={handleViewTask}
        onGenerateWorkflow={handleGenerateWorkflow}
        onClose={() => setSuccessDialog({ ...successDialog, open: false })}
      />
    </Container>
  );
}
