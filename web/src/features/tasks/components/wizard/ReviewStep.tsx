/**
 * Stage 4: Review & Finalization.
 * Implements SRS 2.3.1 Stage 4:
 * - Read-only summary of all data
 * - Split layout: Name/Metrics (left), Description (right)
 * - Pipeline tags as flat list
 * - Justified dataset statistics
 */

import {
  Stack,
  Typography,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Alert,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import FolderIcon from "@mui/icons-material/Folder";
import DescriptionIcon from "@mui/icons-material/Description";

import { Tag } from "../../../../components";
import { AVAILABLE_METRICS } from "../../../../config/metrics";
import type { EditableQuery } from "../../../../types/tasks.types";

interface ReviewStepProps {
  name: string;
  description: string;
  metrics: string[];
  pipelineTags: string[];
  queries: EditableQuery[];
  onEditStep: (step: number) => void;
  submitError?: string | null;
}

interface SectionProps {
  title: string;
  onEdit: () => void;
  children: React.ReactNode;
}

function Section({ title, onEdit, children }: SectionProps) {
  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" fontWeight={600}>
          {title}
        </Typography>
        <Tooltip title="Edit this section">
          <IconButton size="small" onClick={onEdit}>
            <EditIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      {children}
    </Paper>
  );
}

interface StatBoxProps {
  value: number;
  label: string;
  color?: string;
}

function StatBox({ value, label, color = "text.primary" }: StatBoxProps) {
  return (
    <Box sx={{ textAlign: "center", flex: 1 }}>
      <Typography variant="h4" fontWeight={600} color={color}>
        {value}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

export default function ReviewStep({
  name,
  description,
  metrics,
  pipelineTags,
  queries,
  onEditStep,
  submitError,
}: ReviewStepProps) {
  const validationQueries = queries.filter((q) => q.split === "validation");
  const testQueries = queries.filter((q) => q.split === "test");
  const totalFiles = queries.reduce((acc, q) => acc + q.files.length, 0);

  return (
    <Stack spacing={3}>
      {/* Header */}
      <Box>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          Review Task
        </Typography>
        <Typography color="text.secondary">
          Review your task configuration before creating.
        </Typography>
      </Box>

      {submitError && <Alert severity="error">{submitError}</Alert>}

      {/* Metadata Section - Split Layout */}
      <Section title="Task Metadata" onEdit={() => onEditStep(0)}>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
            gap: 3,
          }}
        >
          {/* Left: Name and Metrics */}
          <Stack spacing={3}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Task Name
              </Typography>
              <Typography variant="body1" fontWeight={600}>
                {name || <em>Not specified</em>}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                Evaluation Metrics
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {metrics.length > 0 ? (
                  metrics.map((m) => (
                    <Tag
                      key={m}
                      label={AVAILABLE_METRICS.find((am) => am.value === m)?.label || m}
                      variant="selected"
                      size="small"
                    />
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    <em>No metrics selected</em>
                  </Typography>
                )}
              </Box>
            </Box>
          </Stack>

          {/* Right: Description */}
          <Box>
            <Typography variant="caption" color="text.secondary">
              Description
            </Typography>
            <Typography
              variant="body1"
              sx={{
                whiteSpace: "pre-wrap",
                mt: 0.5,
              }}
            >
              {description || <em>No description</em>}
            </Typography>
          </Box>
        </Box>
      </Section>

      {/* Pipeline Section */}
      <Section
        title={`Pipeline Tags (${pipelineTags.length})`}
        onEdit={() => onEditStep(1)}
      >
        <Box display="flex" flexWrap="wrap" gap={1}>
          {pipelineTags.length > 0 ? (
            pipelineTags.map((tag) => (
              <Tag key={tag} label={tag} variant="selected" size="small" />
            ))
          ) : (
            <Typography variant="body2" color="text.secondary">
              <em>No pipeline tags selected</em>
            </Typography>
          )}
        </Box>
      </Section>

      {/* Dataset Section */}
      <Section title="Dataset" onEdit={() => onEditStep(2)}>
        {/* Stats - Justified */}
        <Box
          sx={{
            display: "flex",
            gap: 2,
            mb: 3,
          }}
        >
          <StatBox value={queries.length} label="Total" />
          <StatBox value={validationQueries.length} label="Validation" color="success.main" />
          <StatBox value={testQueries.length} label="Test" color="info.main" />
          <StatBox value={totalFiles} label="Files" />
        </Box>

        {queries.length > 0 ? (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>#</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Files</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Label</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {queries.map((query, index) => (
                  <TableRow key={query.id}>
                    <TableCell>{index + 1}</TableCell>
                    <TableCell>
                      <Tag
                        label={query.split}
                        variant={query.split === "validation" ? "validation" : "test"}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={0.5}>
                        <FolderIcon fontSize="small" color="action" />
                        {query.files.length}
                      </Box>
                    </TableCell>
                    <TableCell>
                      {query.label ? (
                        <Box display="flex" alignItems="center" gap={0.5}>
                          <DescriptionIcon fontSize="small" color="action" />
                          <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                            {query.label}
                          </Typography>
                        </Box>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          —
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary">
            <em>No queries configured</em>
          </Typography>
        )}
      </Section>
    </Stack>
  );
}
