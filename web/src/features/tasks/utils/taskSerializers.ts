import type {
  EditableQuery,
  UpdateTaskPayload,
} from "../../../types/tasks.types";

export function toUpdateTaskPayload(draft: {
  name: string;
  metric: string;
  description: string;
  queries: EditableQuery[];
}): UpdateTaskPayload {
  return {
    name: draft.name,
    metric: draft.metric,
    description: draft.description,
    queries: draft.queries.map((q) => ({
      id: q.id,
      name: q.name || "",
      split: q.split,
      label: q.label,
      files: q.files.map((f) => ({
        filename: f.file.name,
        content_type: f.file.type,
      })),
    })),
  };
}
