import { rawFetch } from "../api/client";
import type { CreateTaskResponse } from "../types/tasks.types";
import type { QueryData } from "../features/tasks/types";

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
  queries: QueryData[]
) {
  const committedFiles: {
    query_index: number;
    filename: string;
    object_key: string;
    content_type: string;
    size: number;
  }[] = [];

  // Build lookup: query_index -> filename -> File
  const fileMap = new Map<number, Map<string, File>>();

  for (const q of queries) {
    const m = new Map<string, File>();
    for (const f of q.files) {
      m.set(f.file.name, f.file);
    }
    fileMap.set(q.id, m);
  }

  // Upload files
  for (const queryUpload of result.uploads) {
    const qFiles = fileMap.get(queryUpload.query_index);
    if (!qFiles) continue;

    for (const fileInfo of queryUpload.files) {
      const file = qFiles.get(fileInfo.filename);
      if (!file) continue;

      await rawFetch(fileInfo.upload_url, {
        method: "PUT",
        body: file,
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
