import type { CreateTaskResponse } from "../types/tasks.types";
import type { QueryData } from "../features/tasks/types";

/**
 * Upload all files for a created task using presigned POST policies.
 *
 * Guarantees:
 * - Uses multipart/form-data (POST)
 * - No JSON headers
 * - Resume-safe (re-run overwrites same object_key)
 */
export async function uploadTaskFiles(
  result: CreateTaskResponse,
  queries: QueryData[]
) {
  const committedFiles = [];

  const fileMap = new Map<number, Map<string, File>>();
  for (const q of queries) {
    const m = new Map<string, File>();
    for (const f of q.files) {
      m.set(f.file.name, f.file);
    }
    fileMap.set(q.id, m);
  }

  for (const queryUpload of result.uploads) {
    const qFiles = fileMap.get(queryUpload.query_index);
    if (!qFiles) continue;

    for (const fileInfo of queryUpload.files) {
      const file = qFiles.get(fileInfo.filename);
      if (!file) continue;

      const form = new FormData();
      form.append("key", fileInfo.object_key);
      form.append("Content-Type", file.type);
      Object.entries(fileInfo.fields).forEach(([k, v]) =>
        form.append(k, v as string)
      );
      form.append("file", file);

      await fetch(fileInfo.url, {
        method: "POST",
        body: form,
      });

      committedFiles.push({
        query_index: queryUpload.query_index,
        filename: file.name,
        object_key: fileInfo.object_key,
        content_type: file.type,
        size: file.size,
      });
    }
  }

  return committedFiles;
}

