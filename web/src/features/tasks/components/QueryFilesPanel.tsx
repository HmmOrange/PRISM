import { Box, Stack, Divider, Button, Tooltip } from "@mui/material";
import { useState } from "react";
import FileItem from "./FileItem";
import FilePreview from "./FilePreview";

import type {
  EditableQuery,
  QueryDetail,
  QueryFile,
  LocalQueryFile,
} from "../../../types/tasks.types";

type AnyQueryFile = QueryFile | LocalQueryFile;

function isLocalFile(f: AnyQueryFile): f is LocalQueryFile {
  return "file" in f;
}

function getFileKey(f: AnyQueryFile): string {
  return isLocalFile(f) ? f.id : f.object_key;
}

function formatSize(bytes?: number) {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileMeta(file?: AnyQueryFile) {
  if (!file) {
    return {
      type: "",
      size: undefined,
      previewFile: null as File | null,
      downloadUrl: undefined as string | undefined,
    };
  }

  // Create / Edit mode
  if (isLocalFile(file)) {
    return {
      type: file.file.type || "unknown",
      size: file.file.size,
      previewFile: file.file,
      downloadUrl: undefined,
    };
  }

  // View mode (persisted, PROXY ONLY)
  return {
    type: file.content_type || "unknown",
    size: file.size,
    previewFile: null,
    downloadUrl: `/storage/download?object_key=${encodeURIComponent(
      file.object_key
    )}&_ts=${Date.now()}`,
  };
}


interface Props {
  query: EditableQuery | QueryDetail;
  onUpdate?: (q: EditableQuery) => void;
  onDeleteFile?: (fileKey: string) => void;
  readOnly?: boolean;
}

export default function QueryFilesPanel({
  query,
  onUpdate,
  onDeleteFile,
  readOnly = false,
}: Props) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const selectedFile = query.files.find(
    (f) => getFileKey(f) === selectedKey
  );

  function addFiles(files: FileList) {
    if (!onUpdate) return;
    if (!("files" in query)) return;

    onUpdate({
      ...(query as EditableQuery),
      files: [
        ...(query as EditableQuery).files,
        ...Array.from(files).map((f) => ({
          id: crypto.randomUUID(),
          file: f,
        })),
      ],
    });
  }

  const selectedMeta = getFileMeta(selectedFile);

  return (
    <Box display="flex" minHeight={260} border="1px solid" borderColor="divider">
      {/* ===== File list ===== */}
      <Box width={280} p={1} overflow="auto">
        <Stack spacing={0.5}>
          {query.files.map((f) => {
            const key = getFileKey(f);
            const meta = getFileMeta(f);

            return (
              <Tooltip
                key={key}
                title={`${meta.type || "unknown"} · ${formatSize(meta.size)}`}
              >
                <div>
                  <FileItem
                    file={f}
                    selected={key === selectedKey}
                    onSelect={() => setSelectedKey(key)}
                    onDelete={
                      readOnly ? undefined : () => onDeleteFile?.(key)
                    }
                    readOnly={readOnly}
                  />
                </div>
              </Tooltip>
            );
          })}

          {!readOnly && (
            <Button component="label" size="small">
              Add files
              <input
                hidden
                multiple
                type="file"
                onChange={(e) =>
                  e.target.files && addFiles(e.target.files)
                }
              />
            </Button>
          )}
        </Stack>
      </Box>

      <Divider orientation="vertical" flexItem />

      {/* ===== Preview ===== */}
      <Box
        flex={1}
        p={2}
        display="flex"
        alignItems="center"
        justifyContent="center"
        overflow="auto"
        bgcolor="background.default"
      >
        <FilePreview
          file={selectedMeta.previewFile}
          downloadUrl={selectedMeta.downloadUrl}
          contentType={selectedMeta.type}
        />
      </Box>
    </Box>
  );
}
