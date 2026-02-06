/**
 * Wizard Navigation component.
 * Back/Next buttons for the wizard.
 */

import { Box, Button, CircularProgress } from "@mui/material";

interface WizardNavigationProps {
  activeStep: number;
  totalSteps: number;
  canProceed: boolean;
  isSubmitting?: boolean;
  onBack: () => void;
  onNext: () => void;
  onSubmit?: () => void;
  submitLabel?: string;
}

export default function WizardNavigation({
  activeStep,
  totalSteps,
  canProceed,
  isSubmitting = false,
  onBack,
  onNext,
  onSubmit,
  submitLabel = "Create Task",
}: WizardNavigationProps) {
  const isFirstStep = activeStep === 0;
  const isLastStep = activeStep === totalSteps - 1;

  function handleNextOrSubmit() {
    if (isLastStep && onSubmit) {
      onSubmit();
    } else {
      onNext();
    }
  }

  return (
    <Box
      display="flex"
      justifyContent="space-between"
      mt={4}
      pt={3}
      borderTop={1}
      borderColor="divider"
    >
      <Button
        onClick={onBack}
        disabled={isFirstStep || isSubmitting}
        sx={{ minWidth: 100 }}
      >
        Back
      </Button>

      <Button
        variant="contained"
        onClick={handleNextOrSubmit}
        disabled={!canProceed || isSubmitting}
        sx={{ minWidth: 140 }}
      >
        {isSubmitting ? (
          <CircularProgress size={24} color="inherit" />
        ) : isLastStep ? (
          submitLabel
        ) : (
          "Next"
        )}
      </Button>
    </Box>
  );
}
