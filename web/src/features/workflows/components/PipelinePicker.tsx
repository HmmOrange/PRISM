import {
  Box,
  FormControl,
  FormLabel,
  RadioGroup,
  Radio,
  FormControlLabel,
  TextField,
  Typography,
} from "@mui/material";

import type { RunMode } from "../hooks/useRunWorkflowWizard";

interface Props {
  pipelinePath: string | null;
  runMode: RunMode;
  rounds: number;

  onPipelineChange: (path: string) => void;
  onRunModeChange: (mode: RunMode) => void;
  onRoundsChange: (rounds: number) => void;
}

export default function PipelinePicker({
  pipelinePath,
  runMode,
  rounds,
  onPipelineChange,
  onRunModeChange,
  onRoundsChange,
}: Props) {
  return (
    <Box display="flex" flexDirection="column" gap={4}>
      <Typography variant="h6">
        Configure Workflow Run
      </Typography>

      {/* Pipeline */}
      <TextField
        label="Pipeline path"
        placeholder="pipeline/fewshot_pipeline.py"
        value={pipelinePath ?? ""}
        onChange={(e) => onPipelineChange(e.target.value)}
        fullWidth
      />

      {/* Run mode */}
      <FormControl>
        <FormLabel>Run mode</FormLabel>
        <RadioGroup
          value={runMode}
          onChange={(e) =>
            onRunModeChange(e.target.value as RunMode)
          }
        >
          <FormControlLabel
            value="GENERATE_ONLY"
            control={<Radio />}
            label="Generate workflows only"
          />
          <FormControlLabel
            value="GENERATE_AND_RUN"
            control={<Radio />}
            label="Generate and run workflows"
          />
          <FormControlLabel
            value="GENERATE_RUN_AND_SCORE"
            control={<Radio />}
            label="Generate, run, and calculate scores"
          />
        </RadioGroup>
      </FormControl>

      {/* Rounds */}
      <TextField
        label="Rounds"
        type="number"
        value={rounds}
        inputProps={{ min: 1 }}
        onChange={(e) => onRoundsChange(Number(e.target.value))}
        sx={{ width: 200 }}
      />
    </Box>
  );
}
