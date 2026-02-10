import {
  Container,
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  Stack,
  IconButton,
  Tooltip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
} from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import EditIcon from "@mui/icons-material/Edit";
import FolderIcon from "@mui/icons-material/Folder";
import DescriptionIcon from "@mui/icons-material/Description";

import ImportTaskTab from "../components/ImportTaskTab";
import EditSectionModal from "../components/EditSectionModal";
import { MetadataStep, PipelineStep, DatasetStep } from "../components/wizard";

import { Tag } from "../../../components";
import type { EditableQuery } from "../../../types/tasks.types";
import {
  commitTaskFiles,
  createTask,
  importTaskFromZip,
} from "../../../api/tasks.api";
import { uploadTaskFiles } from "../../../utils/uploadExecutor";
import { useToast } from "../../../components/feedback/ToastProvider";
import { ROUTES } from "../../../config/routes";
import { AVAILABLE_METRICS } from "../../../config/metrics";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface ValidationErrors {
  name?: string;
  metrics?: string;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      sx={{ py: 3 }}
    >
      {value === index && children}
    </Box>
  );
}

/** Section wrapper with edit button - matches ReviewStep design */
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

/** Stat box component - matches ReviewStep design */
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

type ModalType = "metadata" | "pipeline" | "dataset" | null;

export default function CreateTaskPage() {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [activeTab, setActiveTab] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState<ValidationErrors>({});

  // Active modal for editing
  const [activeModal, setActiveModal] = useState<ModalType>(null);

  // Create task form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [metrics, setMetrics] = useState<string[]>([]);
  const [pipelineTags, setPipelineTags] = useState<string[]>([]);
  const [queries, setQueries] = useState<EditableQuery[]>([]);

  // Temporary state for editing in modals (to support cancel)
  const [tempName, setTempName] = useState("");
  const [tempDescription, setTempDescription] = useState("");
  const [tempMetrics, setTempMetrics] = useState<string[]>([]);
  const [tempPipelineTags, setTempPipelineTags] = useState<string[]>([]);
  const [tempQueries, setTempQueries] = useState<EditableQuery[]>([]);

  async function handleZipUpload(file: File) {
    if (submitting) return;

    setSubmitting(true);
    try {
      const task = await importTaskFromZip(file);
      showToast({ message: `Task "${task.name}" imported successfully`, severity: "success" });
      navigate(ROUTES.public.tasks);
    } catch (err) {
      console.error("ZIP import failed", err);
      showToast({ message: "Failed to import task from ZIP", severity: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  function validateForm(): boolean {
    const newErrors: ValidationErrors = {};
    
    if (!name.trim()) {
      newErrors.name = "Task name is required";
    }
    if (metrics.length === 0) {
      newErrors.metrics = "Please select at least one metric";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleCreateTask() {
    if (submitting) return;

    // Validate first
    if (!validateForm()) {
      showToast({ message: "Please fill in all required fields", severity: "warning" });
      return;
    }

    setSubmitting(true);
    try {
      const payload = {
        name: name,
        metric: metrics[0] || "", // Primary metric
        description: description,
        queries: queries.map((q) => ({
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

      const hasFiles = queries.some((q) => q.files.length > 0);
      if (hasFiles) {
        // Upload files AFTER task is created
        const committedFiles = await uploadTaskFiles(result, queries);

        // Commit metadata
        await commitTaskFiles(result.task_id, committedFiles);
      }

      showToast({ message: `Task "${name}" created successfully`, severity: "success" });
      navigate(ROUTES.public.tasks);
    } catch (err) {
      console.error("Task creation or upload failed", err);
      showToast({ message: "Failed to create task", severity: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  // Modal handlers
  function openModal(type: ModalType) {
    // Copy current values to temp state
    setTempName(name);
    setTempDescription(description);
    setTempMetrics([...metrics]);
    setTempPipelineTags([...pipelineTags]);
    setTempQueries(queries.map(q => ({ ...q, files: [...q.files] })));
    setActiveModal(type);
  }

  function closeModal() {
    setActiveModal(null);
  }

  function saveMetadata() {
    setName(tempName);
    setDescription(tempDescription);
    setMetrics(tempMetrics);
    // Clear errors
    if (tempName.trim()) {
      setErrors((prev) => ({ ...prev, name: undefined }));
    }
    if (tempMetrics.length > 0) {
      setErrors((prev) => ({ ...prev, metrics: undefined }));
    }
    closeModal();
  }

  function savePipeline() {
    setPipelineTags(tempPipelineTags);
    closeModal();
  }

  function saveDataset() {
    setQueries(tempQueries);
    closeModal();
  }

  function toggleTempPipelineTag(tag: string) {
    setTempPipelineTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }

  // Computed stats
  const validationQueries = queries.filter((q) => q.split === "validation");
  const testQueries = queries.filter((q) => q.split === "test");
  const totalFiles = queries.reduce((acc, q) => acc + q.files.length, 0);

  // Check if we have validation errors to display
  const hasErrors = errors.name || errors.metrics;

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        Create New Task
      </Typography>
      <Typography color="text.secondary" mb={3}>
        Configure your task details. Click the edit button on each section to modify.
      </Typography>

      <Paper elevation={2} sx={{ mt: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            variant="fullWidth"
          >
            <Tab label="Import from ZIP" />
            <Tab label="Create Manually" />
          </Tabs>
        </Box>

        <Box sx={{ p: 3 }}>
          <TabPanel value={activeTab} index={0}>
            <ImportTaskTab
              onZipUpload={handleZipUpload}
              disabled={submitting}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Stack spacing={3}>
              {/* Validation Errors Alert */}
              {hasErrors && (
                <Alert severity="warning">
                  {errors.name && <div>{errors.name}</div>}
                  {errors.metrics && <div>{errors.metrics}</div>}
                </Alert>
              )}

              {/* Metadata Section */}
              <Section title="Task Metadata" onEdit={() => openModal("metadata")}>
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
                        {name || <em style={{ fontWeight: 400, color: "#999" }}>Not specified</em>}
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
                      {description || <em style={{ color: "#999" }}>No description</em>}
                    </Typography>
                  </Box>
                </Box>
              </Section>

              {/* Pipeline Section */}
              <Section
                title={`Pipeline Tags (${pipelineTags.length})`}
                onEdit={() => openModal("pipeline")}
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
              <Section title="Dataset" onEdit={() => openModal("dataset")}>
                {/* Stats */}
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

              {/* Submit Button */}
              <Box display="flex" justifyContent="flex-end" pt={2}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleCreateTask}
                  disabled={submitting}
                >
                  {submitting ? "Creating..." : "Create Task"}
                </Button>
              </Box>
            </Stack>
          </TabPanel>
        </Box>
      </Paper>

      {/* Metadata Edit Modal */}
      <EditSectionModal
        open={activeModal === "metadata"}
        title="Edit Task Metadata"
        onClose={closeModal}
        onSave={saveMetadata}
      >
        <MetadataStep
          name={tempName}
          description={tempDescription}
          metrics={tempMetrics}
          errors={{}}
          onNameChange={setTempName}
          onDescriptionChange={setTempDescription}
          onMetricsChange={setTempMetrics}
        />
      </EditSectionModal>

      {/* Pipeline Edit Modal */}
      <EditSectionModal
        open={activeModal === "pipeline"}
        title="Edit Pipeline Tags"
        onClose={closeModal}
        onSave={savePipeline}
        maxWidth="lg"
      >
        <PipelineStep
          selectedTags={tempPipelineTags}
          onToggleTag={toggleTempPipelineTag}
        />
      </EditSectionModal>

      {/* Dataset Edit Modal */}
      <EditSectionModal
        open={activeModal === "dataset"}
        title="Edit Dataset"
        onClose={closeModal}
        onSave={saveDataset}
        maxWidth="lg"
      >
        <DatasetStep
          queries={tempQueries}
          onQueriesChange={setTempQueries}
        />
      </EditSectionModal>
    </Container>
  );
}
