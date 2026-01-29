import { rawFetch } from "../api/client";
import type { CreateTaskResponse } from "../api/tasks.api";

/**
 * Upload all files for a created task using presigned URLs.
 *
 * Guarantees:
 * - No JSON headers
 * - PUT uploads only
 * - Resume-safe (re-run = overwrite same object_key)
 */
export async function uploadTaskFiles(
  result: CreateTaskResponse,
  queries: {
    id: number;
    files: {
      file: File;
    }[];
  }[]
) {
  // Build lookup: query_id -> filename -> File
  const fileMap = new Map<number, Map<string, File>>();

  for (const q of queries) {
    const m = new Map<string, File>();
    for (const f of q.files) {
      m.set(f.file.name, f.file);
    }
    fileMap.set(q.id, m);
  }

  // Upload sequentially per file (safe default)
  for (const queryUpload of result.uploads) {
    const qFiles = fileMap.get(queryUpload.query_id);
    if (!qFiles) continue;

    for (const fileInfo of queryUpload.files) {
      const file = qFiles.get(fileInfo.filename);
      if (!file) continue;

      await rawFetch(fileInfo.upload_url, {
        method: "PUT",
        body: file,
      });
    }
  }
}
