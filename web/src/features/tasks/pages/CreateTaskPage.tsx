import { Container, Stack, Button, Divider } from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import SectionCard from "../components/SectionCard";
import TaskMetaForm from "../components/TaskMetaForm";
import DatasetEditor from "../components/DatasetEditor";

import type { QueryData } from "../types";
import {
  commitTaskFiles,
  createTask,
  importTaskFromZip,
} from "../../../api/tasks.api";
import { uploadTaskFiles } from "../../../utils/uploadExecutor";
import { useToast } from "../../../components/feedback/ToastProvider";

export default function CreateTaskPage() {
  const navigate = useNavigate();

  const [meta, setMeta] = useState({
    name: "",
    metric: "",
    description: "",
  });

  const [queries, setQueries] = useState<QueryData[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const { showToast } = useToast();

  async function handleZipUpload(file: File) {
    if (submitting) return;

    setSubmitting(true);
    try {
      const task = await importTaskFromZip(file);
      console.log(`Task ${task.name} imported from ZIP`);
      navigate(`/tasks`);
      showToast({ message: `Task "${task.name}" imported`, severity: "success" });
    } catch (err) {
      console.error("ZIP import failed", err);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCreateTask() {
    if (submitting) return;

    setSubmitting(true);
    try {
      const payload = {
        name: meta.name,
        metric: meta.metric,
        description: meta.description,
        queries: queries.map((q) => ({
          id: q.id,
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
      if (!hasFiles) {
        console.log("Task created (no files)");
        return;
      }

      // Upload files AFTER task is created
      const committedFiles = await uploadTaskFiles(result, queries);

      // Commit metadata
      await commitTaskFiles(result.task_id, committedFiles);

      console.log("Task created and files uploaded");
    } catch (err) {
      console.error("Task creation or upload failed", err);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <Stack spacing={4}>
        {/* ===== ZIP Import ===== */}
        <SectionCard title="Import Task">
          <Button
            variant="outlined"
            component="label"
            disabled={submitting}
          >
            Upload ZIP File
            <input
              type="file"
              accept=".zip"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  handleZipUpload(file);
                  e.target.value = "";
                }
              }}
            />
          </Button>
        </SectionCard>

        <Divider />

        {/* ===== Metadata ===== */}
        <SectionCard title="Task Metadata">
          <TaskMetaForm
            name={meta.name}
            metric={meta.metric}
            description={meta.description}
            onChange={(field, value) =>
              setMeta({ ...meta, [field]: value })
            }
          />
        </SectionCard>

        {/* ===== Dataset ===== */}
        <SectionCard title="Dataset">
          <DatasetEditor queries={queries} setQueries={setQueries} />
        </SectionCard>

        {/* ===== Submit ===== */}
        <Button
          variant="contained"
          size="large"
          onClick={handleCreateTask}
          disabled={submitting}
        >
          {submitting ? "Creating..." : "Create Task"}
        </Button>
      </Stack>
    </Container>
  );
}
