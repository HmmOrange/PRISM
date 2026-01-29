import { Box, Stack, Divider, Button, Tooltip } from "@mui/material";
import { useState } from "react";
import FileItem from "./FileItem";
import FilePreview from "./FilePreview";
import type { QueryData } from "../types";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface Props {
  query: QueryData;
  onUpdate: (q: QueryData) => void;
  onDeleteFile: (fileId: string) => void;
}

export default function QueryFilesPanel({
  query,
  onUpdate,
  onDeleteFile,
}: Props) {
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const selectedFile = query.files.find((f) => f.id === selectedFileId);

  function addFiles(files: FileList) {
    onUpdate({
      ...query,
      files: [
        ...query.files,
        ...Array.from(files).map((f) => ({
          id: crypto.randomUUID(),
          file: f,
        })),
      ],
    });
  }

  return (
    <Box display="flex" minHeight={260} border="1px solid" borderColor="divider">
      <Box width={280} p={1} overflow="auto">
        <Stack spacing={0.5}>
          {query.files.map((f) => (
            <Tooltip
              key={f.id}
              title={`${f.file.type || "unknown"} · ${formatSize(
                f.file.size
              )}`}
            >
              <div>
                <FileItem
                  file={f}
                  selected={f.id === selectedFileId}
                  onSelect={() => setSelectedFileId(f.id)}
                  onDelete={() => onDeleteFile(f.id)}
                />
              </div>
            </Tooltip>
          ))}

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
        </Stack>
      </Box>

      <Divider orientation="vertical" flexItem />

      <Box
        flex={1}
        p={2}
        display="flex"
        alignItems="center"
        justifyContent="center"
        overflow="auto"
        bgcolor="background.default"
      >
        <FilePreview file={selectedFile?.file} />
      </Box>
    </Box>
  );
}
