import { Box, Button, Typography, Stack, Paper, LinearProgress } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import FolderZipIcon from "@mui/icons-material/FolderZip";
import { useState, useRef } from "react";

interface Props {
  onZipUpload: (file: File) => Promise<void>;
  disabled?: boolean;
}

export default function ImportTaskTab({ onZipUpload, disabled = false }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      return;
    }
    setUploading(true);
    try {
      await onZipUpload(file);
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
  }

  return (
    <Stack spacing={3}>
      <Typography variant="body2" color="text.secondary">
        Upload a ZIP file containing your task. The ZIP should follow the standard task structure
        with <code>task_description.txt</code>, <code>metadata.json</code>, and <code>test/</code> 
        and <code>validation/</code> folders.
      </Typography>

      <Paper
        elevation={dragOver ? 4 : 1}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        sx={{
          p: 6,
          textAlign: "center",
          border: "2px dashed",
          borderColor: dragOver ? "primary.main" : "divider",
          bgcolor: dragOver ? "action.hover" : "background.paper",
          cursor: disabled || uploading ? "not-allowed" : "pointer",
          transition: "all 0.2s ease",
          "&:hover": {
            borderColor: "primary.light",
            bgcolor: "action.hover",
          },
        }}
        onClick={() => !disabled && !uploading && fileInputRef.current?.click()}
      >
        <FolderZipIcon sx={{ fontSize: 48, color: "primary.main", mb: 2 }} />
        
        <Typography variant="h6" gutterBottom>
          {uploading ? "Uploading..." : "Drop your ZIP file here"}
        </Typography>
        
        <Typography variant="body2" color="text.secondary" gutterBottom>
          or click to browse
        </Typography>

        {uploading && (
          <Box sx={{ mt: 2, width: "50%", mx: "auto" }}>
            <LinearProgress />
          </Box>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".zip"
          hidden
          disabled={disabled || uploading}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              handleFile(file);
              e.target.value = "";
            }
          }}
        />
      </Paper>

      <Box display="flex" justifyContent="center">
        <Button
          variant="outlined"
          component="label"
          startIcon={<CloudUploadIcon />}
          disabled={disabled || uploading}
        >
          Select ZIP File
          <input
            type="file"
            accept=".zip"
            hidden
            disabled={disabled || uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleFile(file);
                e.target.value = "";
              }
            }}
          />
        </Button>
      </Box>

      {/* Help section */}
      <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Expected ZIP structure:
        </Typography>
        <Box
          component="pre"
          sx={{
            fontSize: "0.8rem",
            bgcolor: "grey.100",
            p: 1.5,
            borderRadius: 1,
            overflow: "auto",
            m: 0,
          }}
        >
{`task_folder/
├── task_description.txt
├── metadata.json    # { "metric": "accuracy" }
├── test/
│   ├── labels.csv   # id,label
│   └── inputs/
│       ├── 0/       # Query files
│       └── 1/
└── validation/
    ├── labels.csv
    └── inputs/
        ├── 0/
        └── 1/`}
        </Box>
      </Paper>
    </Stack>
  );
}
