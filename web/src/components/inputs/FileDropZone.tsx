/**
 * File Drop Zone component.
 * Implements SRS 3.1: Drag & Drop file upload functionality.
 */

import { useState, useCallback, useRef } from "react";
import { Box, Typography, IconButton, Stack, Chip } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DeleteIcon from "@mui/icons-material/Delete";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";

interface FileDropZoneProps {
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // in bytes
  maxFiles?: number;
  files: File[];
  onFilesChange: (files: File[]) => void;
  disabled?: boolean;
  helperText?: string;
}

// Blocked extensions
const BLOCKED_EXTENSIONS = [".exe", ".bat", ".cmd", ".sh", ".msi"];

export default function FileDropZone({
  accept = "*/*",
  multiple = true,
  maxSize = 50 * 1024 * 1024, // 50MB default per SRS 3.4
  maxFiles = 10,
  files,
  onFilesChange,
  disabled = false,
  helperText,
}: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /**
   * Validate file before adding.
   */
  function validateFile(file: File): string | null {
    // Check file size
    if (file.size > maxSize) {
      return `File "${file.name}" exceeds maximum size of ${formatFileSize(maxSize)}`;
    }

    // Check blocked extensions
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (BLOCKED_EXTENSIONS.includes(ext)) {
      return `File type "${ext}" is not allowed`;
    }

    return null;
  }

  /**
   * Handle files being added.
   */
  const handleFiles = useCallback(
    (newFiles: FileList | File[]) => {
      setError(null);

      const fileArray = Array.from(newFiles);
      const validFiles: File[] = [];

      // Validate each file
      for (const file of fileArray) {
        const error = validateFile(file);
        if (error) {
          setError(error);
          continue;
        }
        validFiles.push(file);
      }

      // Check max files limit
      const totalFiles = files.length + validFiles.length;
      if (totalFiles > maxFiles) {
        setError(`Maximum ${maxFiles} files allowed`);
        return;
      }

      if (validFiles.length > 0) {
        if (multiple) {
          onFilesChange([...files, ...validFiles]);
        } else {
          onFilesChange([validFiles[0]]);
        }
      }
    },
    [files, multiple, maxFiles, maxSize, onFilesChange]
  );

  /**
   * Handle drag events.
   */
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (!disabled && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [disabled, handleFiles]
  );

  /**
   * Handle file input change.
   */
  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
      // Reset input so same file can be selected again
      e.target.value = "";
    }
  }

  /**
   * Remove a file.
   */
  function removeFile(index: number) {
    onFilesChange(files.filter((_, i) => i !== index));
  }

  /**
   * Format file size for display.
   */
  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  return (
    <Box>
      {/* Drop Zone */}
      <Box
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        sx={{
          border: 2,
          borderStyle: "dashed",
          borderColor: isDragging
            ? "primary.main"
            : error
            ? "error.main"
            : "divider",
          borderRadius: 2,
          p: 4,
          textAlign: "center",
          bgcolor: isDragging ? "action.hover" : "background.paper",
          cursor: disabled ? "default" : "pointer",
          opacity: disabled ? 0.5 : 1,
          transition: "all 0.2s",
          "&:hover": {
            bgcolor: disabled ? undefined : "action.hover",
          },
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleInputChange}
          disabled={disabled}
          style={{ display: "none" }}
        />

        <CloudUploadIcon
          sx={{
            fontSize: 48,
            color: isDragging ? "primary.main" : "text.secondary",
            mb: 1,
          }}
        />
        <Typography variant="body1" gutterBottom>
          {isDragging
            ? "Drop files here"
            : "Drag & drop files here or click to browse"}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {helperText ||
            `Max ${formatFileSize(maxSize)} per file${
              multiple ? `, up to ${maxFiles} files` : ""
            }`}
        </Typography>
      </Box>

      {/* Error Message */}
      {error && (
        <Typography variant="caption" color="error" sx={{ mt: 1, display: "block" }}>
          {error}
        </Typography>
      )}

      {/* File List */}
      {files.length > 0 && (
        <Stack spacing={1} mt={2}>
          {files.map((file, index) => (
            <Box
              key={`${file.name}-${index}`}
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                p: 1,
                borderRadius: 1,
                bgcolor: "action.hover",
              }}
            >
              <InsertDriveFileIcon color="action" fontSize="small" />
              <Box flex={1} minWidth={0}>
                <Typography variant="body2" noWrap>
                  {file.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatFileSize(file.size)}
                </Typography>
              </Box>
              <Chip
                size="small"
                label={file.type || "Unknown"}
                variant="outlined"
              />
              <IconButton
                size="small"
                onClick={() => removeFile(index)}
                disabled={disabled}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  );
}
