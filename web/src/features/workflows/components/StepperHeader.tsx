import {
  Stepper,
  Step,
  StepLabel,
  Box,
} from "@mui/material";

const STEPS = [
  "Select Tasks",
  "Configure Run",
  "Running",
  "Results",
];

interface Props {
  activeStep: number;
}

export default function StepperHeader({ activeStep }: Props) {
  return (
    <Box sx={{ width: "100%" }}>
      <Stepper activeStep={activeStep} alternativeLabel>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
    </Box>
  );
}
