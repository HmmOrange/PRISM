import {
  Container,
  Typography,
  Stack,
  CircularProgress,
  Box,
  Button,
  Chip,
} from "@mui/material";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

import { getTask } from "../../../api/tasks.api";
import type {
  TaskDetail,
  EditableQuery,
} from "../../../types/tasks.types";

import SectionCard from "../components/SectionCard";
import QueryAccordion from "../components/QueryAccordion";
import TaskMetaForm from "../components/TaskMetaForm";

import { updateTask } from "../../../api/tasks.api";
import { toUpdateTaskPayload } from "../utils/taskSerializers";
import { apiFetch } from "../../../api/client";
import { useToast } from "../../../components/feedback/ToastProvider";
import { getMetricLabel } from "../../../config/metrics";

type DraftTask = {
  name: string;
  metric: string;
  description: string;
  queries: EditableQuery[];
};

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();

  const [task, setTask] = useState<TaskDetail | null>(null);
  const [mode, setMode] = useState<"view" | "edit">("view");
  const [draft, setDraft] = useState<DraftTask | null>(null);
  const { showToast } = useToast();

  useEffect(() => {
    if (!taskId) return;
    getTask(taskId).then(setTask);
  }, [taskId]);

  function enterEditMode() {
    if (!task) return;

    setDraft({
      name: task.name,
      metric: task.metric,
      description: task.description,
      queries: task.queries.map((q) => ({
        id: q.index,
        name: "",
        split: q.split,
        label: q.label,
        files: [],
      })),
    });

    setMode("edit");
  }

  function cancelEdit() {
    setDraft(null);
    setMode("view");
  }

  function updateQuery(updated: EditableQuery) {
    if (!draft) return;

    setDraft({
      ...draft,
      queries: draft.queries.map((q) =>
        q.id === updated.id ? updated : q
      ),
    });
  }

  function deleteQueryFile(queryId: number, fileId: string) {
    if (!draft) return;

    setDraft({
      ...draft,
      queries: draft.queries.map((q) =>
        q.id !== queryId
          ? q
          : {
              ...q,
              files: q.files.filter((f) => f.id !== fileId),
            }
      ),
    });
  }

  async function saveChanges() {
    if (!draft || !taskId) return;

    const payload = toUpdateTaskPayload(draft);

    // 1. Update task + get presigned uploads
    const res = await updateTask(taskId, payload);

    // 2. Upload new files (reuse existing create flow logic)
    for (const upload of res.uploads) {
      const query = draft.queries.find(
        (q) => q.id === upload.query_index
      );
      if (!query) continue;

      for (let i = 0; i < upload.files.length; i++) {
        const f = upload.files[i];
        const local = query.files[i];

        const formData = new FormData();
        Object.entries(f.fields).forEach(([k, v]) =>
          formData.append(k, v)
        );
        formData.append("file", local.file);

        console.log(f.url, formData);
        await fetch(f.url, {
          method: "POST",
          body: formData,
        });
      }
    }

    // 3. Commit file metadata
    await apiFetch(`/tasks/${taskId}/files/commit`, {
      method: "POST",
      body: JSON.stringify({
        files: res.uploads.flatMap((u) => {
          const query = draft.queries.find(
            (q) => q.id === u.query_index
          );
          if (!query) return [];

          return u.files.map((f, i) => {
            const local = query.files[i];

            return {
              query_index: u.query_index,
              object_key: f.object_key,
              filename: f.filename,
              content_type: local.file.type,
              size: local.file.size,
            };
          });
        }),
      }),
    });

    // 4. Refresh + exit edit mode
    const updated = await getTask(taskId);
    setTask(updated);
    setDraft(null);
    setMode("view");

    showToast({ message: `Task "${draft.name}" updated`, severity: "success" });
  }

  if (!task) {
    return (
      <Container sx={{ mt: 6, display: "flex", justifyContent: "center" }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container sx={{ mt: 4, mb: 6 }}>
      <Stack spacing={4}>
        {/* ===== Header actions ===== */}
        <Stack direction="row" justifyContent="flex-end" spacing={1}>
          {mode === "view" ? (
            <Button variant="outlined" onClick={enterEditMode}>
              Edit
            </Button>
          ) : (
            <>
              <Button onClick={cancelEdit}>Cancel</Button>
              <Button variant="contained" onClick={saveChanges}>
                Save
              </Button>
            </>
          )}
        </Stack>

        {/* ===== Task Metadata ===== */}
        <SectionCard title="Task Metadata">
          {mode === "view" ? (
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle2" color="text.secondary">
                  Name
                </Typography>
                <Typography>{task.name}</Typography>
              </Box>

              <Box>
                <Typography variant="subtitle2" color="text.secondary">
                  Metric
                </Typography>
                <Chip
                  label={getMetricLabel(task.metric)}
                  size="small"
                  color="primary"
                  variant="outlined"
                  sx={{ mt: 0.5 }}
                />
              </Box>

              <Box>
                <Typography variant="subtitle2" color="text.secondary">
                  Description
                </Typography>
                <Typography whiteSpace="pre-line">
                  {task.description || "—"}
                </Typography>
              </Box>
            </Stack>
          ) : (
            <TaskMetaForm
              name={draft!.name}
              metric={draft!.metric}
              description={draft!.description}
              onChange={(field, value) =>
                setDraft({ ...draft!, [field]: value })
              }
            />
          )}
        </SectionCard>

        {/* ===== Dataset ===== */}
        <SectionCard title="Dataset">
          <Stack spacing={2}>
            {(mode === "view" ? task.queries : draft!.queries).map((q) => {
              // Use split + index as unique key to avoid duplicates
              const uniqueKey = "index" in q
                ? `${q.split}-${q.index}`
                : `${q.split}-${q.id}`;

              return (
                <QueryAccordion
                  key={uniqueKey}
                  query={q as any}
                  mode={mode}
                  onUpdate={
                    mode === "edit"
                      ? (uq) => updateQuery(uq)
                      : undefined
                  }
                  onDeleteFile={
                    mode === "edit"
                      ? (fileId) =>
                          deleteQueryFile(
                            (q as EditableQuery).id,
                            fileId
                          )
                      : undefined
                  }
                />
              );
            })}
          </Stack>
        </SectionCard>
      </Stack>
    </Container>
  );
}
