import {
  Container,
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
} from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import ImportTaskTab from "../components/ImportTaskTab";
import CreateTaskTab from "../components/CreateTaskTab";

import type { EditableQuery } from "../../../types/tasks.types";
import {
  commitTaskFiles,
  createTask,
  importTaskFromZip,
} from "../../../api/tasks.api";
import { uploadTaskFiles } from "../../../utils/uploadExecutor";
import { useToast } from "../../../components/feedback/ToastProvider";
import { ROUTES } from "../../../config/routes";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface ValidationErrors {
  name?: string;
  metric?: string;
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

export default function CreateTaskPage() {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [activeTab, setActiveTab] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState<ValidationErrors>({});

  // Create task form state
  const [meta, setMeta] = useState({
    name: "",
    metric: "",
    description: "",
  });
  const [queries, setQueries] = useState<EditableQuery[]>([]);

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
    
    if (!meta.name.trim()) {
      newErrors.name = "Task name is required";
    }
    if (!meta.metric) {
      newErrors.metric = "Please select a metric";
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
        name: meta.name,
        metric: meta.metric,
        description: meta.description,
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

      showToast({ message: `Task "${meta.name}" created successfully`, severity: "success" });
      navigate(ROUTES.public.tasks);
    } catch (err) {
      console.error("Task creation or upload failed", err);
      showToast({ message: "Failed to create task", severity: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  function handleMetaChange(field: string, value: string) {
    setMeta({ ...meta, [field]: value });
    // Clear error when user starts typing
    if (errors[field as keyof ValidationErrors]) {
      setErrors({ ...errors, [field]: undefined });
    }
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      <Typography variant="h4" gutterBottom>
        New Task
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
            <CreateTaskTab
              meta={meta}
              queries={queries}
              onMetaChange={handleMetaChange}
              onQueriesChange={setQueries}
              onSubmit={handleCreateTask}
              submitting={submitting}
              errors={errors}
            />
          </TabPanel>
        </Box>
      </Paper>
    </Container>
  );
}
