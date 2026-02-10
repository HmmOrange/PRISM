import {
  Container,
  Typography,
  Stack,
  CircularProgress,
  Box,
  Paper,
  IconButton,
  Tooltip,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  Tabs,
  Tab,
  Button,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import FolderIcon from "@mui/icons-material/Folder";
import DescriptionIcon from "@mui/icons-material/Description";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CloseIcon from "@mui/icons-material/Close";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import { getTask, updateTask, patchTaskMetadata } from "../../../api/tasks.api";
import type { TaskDetail, EditableQuery, QueryFile } from "../../../types/tasks.types";

import EditSectionModal from "../components/EditSectionModal";
import { MetadataStep, DatasetStep, PipelineStep } from "../components/wizard";
import { Tag } from "../../../components";
import FilePreview from "../components/FilePreview";

import { toUpdateTaskPayload } from "../utils/taskSerializers";
import { apiFetch } from "../../../api/client";
import { useToast } from "../../../components/feedback/ToastProvider";
import { getMetricLabel } from "../../../config/metrics";

/** Section wrapper with edit button - consistent with CreateTaskPage and ReviewStep */
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

type ModalType = "metadata" | "dataset" | "pipeline" | null;

type DraftTask = {
  name: string;
  description: string;
  metrics: string[];
  pipelineTags: string[];
  queries: EditableQuery[];
};

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();

  const [task, setTask] = useState<TaskDetail | null>(null);
  const [activeModal, setActiveModal] = useState<ModalType>(null);
  const [draft, setDraft] = useState<DraftTask | null>(null);
  const [expandedQueries, setExpandedQueries] = useState<Set<number>>(new Set());
  const [previewFile, setPreviewFile] = useState<QueryFile | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);
  const { showToast } = useToast();

  function toggleQueryExpand(index: number) {
    setExpandedQueries((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  useEffect(() => {
    if (!taskId) return;
    getTask(taskId).then(setTask);
  }, [taskId]);

  function openModal(type: ModalType) {
    if (!task) return;

    // Initialize draft with current task values
    setDraft({
      name: task.name,
      description: task.description,
      metrics: [task.metric], // Convert single metric to array for MetadataStep
      pipelineTags: task.pipeline_tags || [],
      queries: task.queries.map((q) => ({
        id: q.index,
        name: q.name || "",
        split: q.split,
        label: q.label,
        files: [], // New files to add
        existingFiles: q.files.map((f) => ({
          id: f.object_key,
          filename: f.filename,
          size: f.size,
          download_url: f.download_url,
          content_type: f.content_type,
          object_key: f.object_key,
        })),
      })),
    });

    setActiveModal(type);
  }

  function closeModal() {
    setActiveModal(null);
    setDraft(null);
  }

  async function saveMetadata() {
    if (!draft || !taskId || !task) return;

    try {
      // Use PATCH for metadata-only updates (preserves files)
      const updated = await patchTaskMetadata(taskId, {
        name: draft.name,
        description: draft.description,
        metric: draft.metrics[0] || task.metric,
      });
      setTask(updated);

      closeModal();
      showToast({ message: `Task "${draft.name}" updated`, severity: "success" });
    } catch (err) {
      console.error("Failed to update task", err);
      showToast({ message: "Failed to update task", severity: "error" });
    }
  }

  async function saveDataset() {
    if (!draft || !taskId || !task) return;

    try {
      const payload = toUpdateTaskPayload({
        name: task.name,
        metric: task.metric,
        description: task.description,
        pipeline_tags: task.pipeline_tags || [],
        queries: draft.queries,
      });

      const res = await updateTask(taskId, payload);

      // Upload new files
      for (const upload of res.uploads) {
        const query = draft.queries.find((q) => q.id === upload.query_index);
        if (!query) continue;

        // Build a map from filename to File for this query
        const fileMap = new Map<string, File>();
        for (const f of query.files) {
          fileMap.set(f.file.name, f.file);
        }

        for (const fileInfo of upload.files) {
          const file = fileMap.get(fileInfo.filename);
          if (!file) continue;

          const formData = new FormData();
          formData.append("key", fileInfo.object_key);
          formData.append("Content-Type", file.type);
          Object.entries(fileInfo.fields).forEach(([k, v]) =>
            formData.append(k, v as string)
          );
          formData.append("file", file);

          await fetch(fileInfo.url, {
            method: "POST",
            body: formData,
          });
        }
      }

      // Commit file metadata
      await apiFetch(`/tasks/${taskId}/files/commit`, {
        method: "POST",
        body: JSON.stringify({
          files: res.uploads.flatMap((u) => {
            const query = draft.queries.find((q) => q.id === u.query_index);
            if (!query) return [];

            // Build a map from filename to File for this query
            const fileMap = new Map<string, File>();
            for (const f of query.files) {
              fileMap.set(f.file.name, f.file);
            }

            return u.files.map((f) => {
              const file = fileMap.get(f.filename);
              if (!file) return null;

              return {
                query_index: u.query_index,
                object_key: f.object_key,
                filename: f.filename,
                content_type: file.type,
                size: file.size,
              };
            }).filter((x): x is NonNullable<typeof x> => x !== null);
          }),
        }),
      });

      // Refresh task data
      const updated = await getTask(taskId);
      setTask(updated);

      closeModal();
      showToast({ message: "Dataset updated", severity: "success" });
    } catch (err) {
      console.error("Failed to update dataset", err);
      showToast({ message: "Failed to update dataset", severity: "error" });
    }
  }

  async function savePipeline() {
    if (!draft || !taskId || !task) return;

    try {
      // Use PATCH for metadata-only updates (preserves files)
      const updated = await patchTaskMetadata(taskId, {
        pipeline_tags: draft.pipelineTags,
      });
      setTask(updated);

      closeModal();
      showToast({ message: "Pipeline tags updated", severity: "success" });
    } catch (err) {
      console.error("Failed to update pipeline tags", err);
      showToast({ message: "Failed to update pipeline tags", severity: "error" });
    }
  }

  if (!task) {
    return (
      <Container sx={{ mt: 6, display: "flex", justifyContent: "center" }}>
        <CircularProgress />
      </Container>
    );
  }

  // Computed stats
  const validationQueries = task.queries.filter((q) => q.split === "validation");
  const testQueries = task.queries.filter((q) => q.split === "test");
  const totalFiles = task.queries.reduce((acc, q) => acc + q.files.length, 0);

  // Format dates
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "—";
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        {task.name}
      </Typography>
      <Box display="flex" alignItems="center" gap={3} mb={3}>
        <Typography color="text.secondary">
          View and edit your task details. Click the edit button on each section to modify.
        </Typography>
        <Box flex={1} />
        <Box display="flex" gap={3}>
          <Box>
            <Typography variant="caption" color="text.secondary" display="block">
              Created
            </Typography>
            <Typography variant="body2">
              {formatDate(task.created_at)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary" display="block">
              Last Modified
            </Typography>
            <Typography variant="body2">
              {formatDate(task.updated_at)}
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          aria-label="Task detail tabs"
        >
          <Tab label="Task Information" />
          <Tab label="Dataset" />
        </Tabs>
      </Box>

      {/* Tab Panel: Task Information */}
      {activeTab === 0 && (
        <Stack spacing={3}>
          {/* Metadata Section */}
          <Section title="Task Metadata" onEdit={() => openModal("metadata")}>
            <Stack spacing={3}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Task Name
                </Typography>
                <Typography
                  variant="body1"
                  fontWeight={600}
                  sx={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                  }}
                  title={task.name}
                >
                  {task.name}
                </Typography>
              </Box>

              <Box>
                <Typography variant="caption" color="text.secondary">
                  Description
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    mt: 0.5,
                    whiteSpace: "pre-wrap",
                    ...(!descriptionExpanded && {
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      display: "-webkit-box",
                      WebkitLineClamp: 5,
                      WebkitBoxOrient: "vertical",
                    }),
                  }}
                >
                  {task.description || <em style={{ color: "#999" }}>No description</em>}
                </Typography>
                {task.description && task.description.split("\n").length > 5 && (
                  <Button
                    size="small"
                    onClick={() => setDescriptionExpanded(!descriptionExpanded)}
                    sx={{ mt: 0.5, p: 0, minWidth: "auto", textTransform: "none" }}
                  >
                    {descriptionExpanded ? "Show less" : "Show more"}
                  </Button>
                )}
              </Box>

              <Box>
                <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                  Evaluation Metrics
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  <Tag
                    label={getMetricLabel(task.metric)}
                    variant="selected"
                    size="small"
                  />
                </Box>
              </Box>
            </Stack>
          </Section>

          {/* Pipeline Tags Section */}
          <Section
            title={`Pipeline Tags (${task.pipeline_tags?.length || 0})`}
            onEdit={() => openModal("pipeline")}
          >
            {task.pipeline_tags && task.pipeline_tags.length > 0 ? (
              <Box display="flex" flexWrap="wrap" gap={1}>
                {task.pipeline_tags.map((tag) => (
                  <Tag key={tag} label={tag} variant="selected" size="small" />
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                <em>No pipeline tags set</em>
              </Typography>
            )}
          </Section>
        </Stack>
      )}

      {/* Tab Panel: Dataset */}
      {activeTab === 1 && (
        <Section title="Dataset" onEdit={() => openModal("dataset")}>
          {/* Stats */}
          <Box
            sx={{
              display: "flex",
              gap: 2,
              mb: 3,
            }}
          >
            <StatBox value={task.queries.length} label="Total" />
            <StatBox value={validationQueries.length} label="Validation" color="success.main" />
            <StatBox value={testQueries.length} label="Test" color="info.main" />
            <StatBox value={totalFiles} label="Files" />
          </Box>

          {task.queries.length > 0 ? (
            <Stack spacing={1}>
              {task.queries.map((query, index) => (
                <Paper
                  key={`${query.split}-${query.index}`}
                  variant="outlined"
                  sx={{
                    overflow: "hidden",
                    transition: "all 0.2s ease-in-out",
                    "&:hover": { borderColor: "text.secondary" },
                  }}
                >
                  {/* Query Header */}
                  <Box
                    onClick={() => toggleQueryExpand(query.index)}
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 2,
                      px: 2,
                      py: 1.5,
                      cursor: "pointer",
                      bgcolor: expandedQueries.has(query.index) ? "action.selected" : "transparent",
                      "&:hover": { bgcolor: "action.hover" },
                    }}
                  >
                    <Typography fontWeight={600}>Query {index + 1}</Typography>
                    <Tag
                      label={query.split}
                      variant={query.split === "validation" ? "validation" : "test"}
                      size="small"
                    />
                    <Box display="flex" alignItems="center" gap={0.5} color="text.secondary">
                      <FolderIcon fontSize="small" />
                      <Typography variant="caption">{query.files.length} files</Typography>
                    </Box>
                    {query.label && (
                      <Box display="flex" alignItems="center" gap={0.5} color="text.secondary">
                        <DescriptionIcon fontSize="small" />
                        <Typography variant="caption" noWrap sx={{ maxWidth: 150 }}>
                          {query.label}
                        </Typography>
                      </Box>
                    )}
                    <Box flex={1} />
                    {expandedQueries.has(query.index) ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </Box>

                  {/* Expanded Content */}
                  <Collapse in={expandedQueries.has(query.index)}>
                    <Box sx={{ p: 2, borderTop: 1, borderColor: "divider" }}>
                      {/* Query Info Summary */}
                      <Box
                        sx={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 3,
                          mb: 2,
                          pb: 2,
                          borderBottom: 1,
                          borderColor: "divider",
                        }}
                      >
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Query Type
                          </Typography>
                          <Box mt={0.5}>
                            <Tag
                              label={query.split}
                              variant={query.split === "validation" ? "validation" : "test"}
                              size="small"
                            />
                          </Box>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Files
                          </Typography>
                          <Typography variant="body1" fontWeight={600} mt={0.5}>
                            {query.files.length}
                          </Typography>
                        </Box>
                        {query.name && (
                          <Box flex={1}>
                            <Typography variant="caption" color="text.secondary">
                              Query Name
                            </Typography>
                            <Typography variant="body2" mt={0.5}>
                              {query.name}
                            </Typography>
                          </Box>
                        )}
                      </Box>

                      {/* Ground Truth Label */}
                      <Box mb={2}>
                        <Typography variant="caption" color="text.secondary">
                          Ground Truth Label
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{ whiteSpace: "pre-wrap", mt: 0.5 }}
                        >
                          {query.label || <em style={{ color: "#999" }}>Not specified</em>}
                        </Typography>
                      </Box>

                      {/* Files Grid */}
                      {query.files.length > 0 ? (
                        <Box
                          sx={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                            gap: 2,
                          }}
                        >
                          {query.files.map((file) => (
                            <Paper
                              key={file.object_key}
                              variant="outlined"
                              onClick={() => setPreviewFile(file)}
                              sx={{
                                p: 1.5,
                                cursor: "pointer",
                                transition: "all 0.2s",
                                "&:hover": {
                                  borderColor: "primary.main",
                                  bgcolor: "action.hover",
                                  transform: "translateY(-2px)",
                                  boxShadow: 2,
                                },
                              }}
                            >
                              {/* Thumbnail/Preview */}
                              <Box
                                sx={{
                                  height: 150,
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  bgcolor: "action.hover",
                                  borderRadius: 1,
                                  mb: 1,
                                  overflow: "hidden",
                                }}
                              >
                                <FilePreview
                                  downloadUrl={file.download_url}
                                  contentType={file.content_type}
                                />
                              </Box>
                              <Typography
                                variant="body2"
                                noWrap
                                display="block"
                                title={file.filename}
                              >
                                {file.filename}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {(file.size / 1024).toFixed(1)} KB
                              </Typography>
                            </Paper>
                          ))}
                        </Box>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          <em>No files in this query</em>
                        </Typography>
                      )}
                    </Box>
                  </Collapse>
                </Paper>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              <em>No queries in this task</em>
            </Typography>
          )}
        </Section>
      )}

      {/* File Preview Modal */}
      <Dialog
        open={!!previewFile}
        onClose={() => setPreviewFile(null)}
        maxWidth="lg"
        fullWidth
      >
        {previewFile && (
          <>
            <DialogTitle sx={{ display: "flex", alignItems: "center" }}>
              <Typography variant="h6" flex={1}>
                {previewFile.filename}
              </Typography>
              <IconButton onClick={() => setPreviewFile(null)}>
                <CloseIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  minHeight: 300,
                }}
              >
                <FilePreview
                  downloadUrl={previewFile.download_url}
                  contentType={previewFile.content_type}
                  showControls
                />
              </Box>
            </DialogContent>
          </>
        )}
      </Dialog>

      {/* Metadata Edit Modal */}
      <EditSectionModal
        open={activeModal === "metadata"}
        title="Edit Task Metadata"
        onClose={closeModal}
        onSave={saveMetadata}
      >
        {draft && (
          <MetadataStep
            name={draft.name}
            description={draft.description}
            metrics={draft.metrics}
            errors={{}}
            onNameChange={(value) => setDraft({ ...draft, name: value })}
            onDescriptionChange={(value) => setDraft({ ...draft, description: value })}
            onMetricsChange={(metrics) => setDraft({ ...draft, metrics })}
          />
        )}
      </EditSectionModal>

      {/* Dataset Edit Modal */}
      <EditSectionModal
        open={activeModal === "dataset"}
        title="Edit Dataset"
        onClose={closeModal}
        onSave={saveDataset}
        maxWidth="lg"
      >
        {draft && (
          <DatasetStep
            queries={draft.queries}
            onQueriesChange={(queries) => setDraft({ ...draft, queries })}
          />
        )}
      </EditSectionModal>

      {/* Pipeline Tags Edit Modal */}
      <EditSectionModal
        open={activeModal === "pipeline"}
        title="Edit Pipeline Tags"
        onClose={closeModal}
        onSave={savePipeline}
        maxWidth="md"
      >
        {draft && (
          <PipelineStep
            selectedTags={draft.pipelineTags}
            onToggleTag={(tag) => {
              const newTags = draft.pipelineTags.includes(tag)
                ? draft.pipelineTags.filter((t) => t !== tag)
                : [...draft.pipelineTags, tag];
              setDraft({ ...draft, pipelineTags: newTags });
            }}
          />
        )}
      </EditSectionModal>
    </Container>
  );
}
