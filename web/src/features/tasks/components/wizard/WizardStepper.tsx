/**
 * Wizard Stepper component.
 * Displays progress through the multi-step wizard.
 * Implements SRS 2.3.1: UI must indicate progress (e.g., "Step 1 of 4")
 */

import {
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  useTheme,
  useMediaQuery,
} from "@mui/material";

interface WizardStepperProps {
  activeStep: number;
  steps: string[];
  onStepClick?: (step: number) => void;
}

export default function WizardStepper({ activeStep, steps, onStepClick }: WizardStepperProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  if (isMobile) {
    // Compact mobile view
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          p: 2,
          bgcolor: "background.paper",
          borderRadius: 1,
          border: 1,
          borderColor: "divider",
        }}
      >
        <Typography variant="subtitle1" fontWeight={600}>
          {steps[activeStep]}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Step {activeStep + 1} of {steps.length}
        </Typography>
      </Box>
    );
  }

  return (
    <Stepper activeStep={activeStep} alternativeLabel>
      {steps.map((label, index) => (
        <Step
          key={label}
          completed={index < activeStep}
          sx={{
            cursor: index <= activeStep && onStepClick ? "pointer" : "default",
          }}
          onClick={() => {
            if (index <= activeStep && onStepClick) {
              onStepClick(index);
            }
          }}
        >
          <StepLabel
            StepIconProps={{
              sx: {
                "&.Mui-active": {
                  color: "primary.main",
                },
                "&.Mui-completed": {
                  color: "success.main",
                },
              },
            }}
          >
            {label}
          </StepLabel>
        </Step>
      ))}
    </Stepper>
  );
}
