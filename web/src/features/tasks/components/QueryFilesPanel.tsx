import { Box, Stack, Divider, Button, Tooltip, Typography, Paper, alpha } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { useState, useCallback } from "react";
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

  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!readOnly) setDragOver(true);
  }, [readOnly]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!readOnly && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  }, [readOnly, addFiles]);

  const selectedMeta = getFileMeta(selectedFile);

  return (
    <Paper
      variant="outlined"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      sx={{
        display: "flex",
        minHeight: 280,
        borderColor: dragOver ? "primary.main" : "divider",
        borderWidth: dragOver ? 2 : 1,
        backgroundColor: dragOver ? (theme) => alpha(theme.palette.primary.main, 0.04) : undefined,
        transition: "all 0.2s ease",
      }}
    >
      {/* ===== File list ===== */}
      <Box width={300} p={1.5} overflow="auto" bgcolor="background.paper">
        <Typography variant="caption" color="text.secondary" fontWeight={500} sx={{ mb: 1, display: "block" }}>
          Files ({query.files.length})
        </Typography>

        <Stack spacing={0.5}>
          {query.files.map((f) => {
            const key = getFileKey(f);
            const meta = getFileMeta(f);

            return (
              <Tooltip
                key={key}
                title={`${meta.type || "unknown"} · ${formatSize(meta.size)}`}
                placement="right"
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

          {query.files.length === 0 && !readOnly && (
            <Box
              sx={{
                py: 3,
                textAlign: "center",
                color: "text.secondary",
              }}
            >
              <CloudUploadIcon sx={{ fontSize: 32, opacity: 0.5, mb: 1 }} />
              <Typography variant="body2">
                Drop files here
              </Typography>
              <Typography variant="caption">
                or click below to browse
              </Typography>
            </Box>
          )}

          {!readOnly && (
            <Button
              component="label"
              size="small"
              variant="outlined"
              startIcon={<CloudUploadIcon />}
              sx={{ mt: 1 }}
            >
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
        bgcolor="grey.50"
      >
        {selectedFile ? (
          <FilePreview
            file={selectedMeta.previewFile}
            downloadUrl={selectedMeta.downloadUrl}
            contentType={selectedMeta.type}
          />
        ) : (
          <Typography variant="body2" color="text.secondary">
            Select a file to preview
          </Typography>
        )}
      </Box>
    </Paper>
  );
}
