/**
 * Stage 1: Metadata Configuration.
 * Implements SRS 2.3.1 Stage 1:
 * - Task Name (Required)
 * - Description (Rich Text/Markdown, Required)
 * - Task Metric (Multi-select, Required)
 */

import { Stack, TextField, Box, Typography } from "@mui/material";

import { FormField, Tag } from "../../../../components";
import { AVAILABLE_METRICS } from "../../../../config/metrics";

interface MetadataStepProps {
  name: string;
  description: string;
  metrics: string[];
  errors?: {
    name?: string;
    metrics?: string;
  };
  onNameChange: (value: string) => void;
  onDescriptionChange: (value: string) => void;
  onMetricsChange: (metrics: string[]) => void;
}

export default function MetadataStep({
  name,
  description,
  metrics,
  errors = {},
  onNameChange,
  onDescriptionChange,
  onMetricsChange,
}: MetadataStepProps) {
  function toggleMetric(metricValue: string) {
    if (metrics.includes(metricValue)) {
      onMetricsChange(metrics.filter((m) => m !== metricValue));
    } else {
      onMetricsChange([...metrics, metricValue]);
    }
  }

  return (
    <Stack spacing={4}>
      {/* Header */}
      <Box>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          Task Metadata
        </Typography>
        <Typography color="text.secondary">
          Define the basic information for your task.
        </Typography>
      </Box>

      {/* Task Name */}
      <FormField
        label="Task Name"
        required
        error={errors.name}
      >
        <TextField
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="Enter a descriptive name for your task"
          error={!!errors.name}
          fullWidth
          autoFocus
          InputLabelProps={{ shrink: false }}
          sx={{ "& .MuiInputBase-input": { py: 1.25 } }}
        />
      </FormField>

      {/* Description */}
      <FormField
        label="Description"
        description="Describe the task objectives and expected outcomes. Supports Markdown."
        required
      >
        <TextField
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder="What is this task about? What are the expected outcomes?"
          multiline
          rows={4}
          fullWidth
          InputLabelProps={{ shrink: false }}
        />
      </FormField>

      {/* Evaluation Metrics */}
      <FormField
        label="Evaluation Metrics"
        description="Select one or more metrics to evaluate task performance."
        required
        error={errors.metrics}
      >
        <Box display="flex" flexWrap="wrap" gap={1}>
          {AVAILABLE_METRICS.map((metric) => (
            <Tag
              key={metric.value}
              label={metric.label}
              variant={metrics.includes(metric.value) ? "selected" : "default"}
              onClick={() => toggleMetric(metric.value)}
            />
          ))}
        </Box>
      </FormField>
    </Stack>
  );
}
