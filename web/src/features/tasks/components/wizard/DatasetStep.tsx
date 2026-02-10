/**
 * Stage 3: Dataset Ingestion.
 * Implements SRS 2.3.1 Stage 3:
 * - Split into Validation and Test dataset sections
 * - Support for manual query creation and ZIP upload
 * - Validation: labels required, recommend 5+ queries
 * - Test: labels optional
 */

import { useState, useRef } from "react";
import {
  Stack,
  Typography,
  Box,
  Button,
  Paper,
  Alert,
  Divider,
  IconButton,
  Tooltip,
  TextField,
  Collapse,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import FolderIcon from "@mui/icons-material/Folder";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

import { Tag, FormField } from "../../../../components";
import FilePreview from "../FilePreview";
import type { EditableQuery, LocalQueryFile } from "../../../../types/tasks.types";

interface DatasetStepProps {
  queries: EditableQuery[];
  onQueriesChange: (queries: EditableQuery[]) => void;
}

// Simple File Drop Zone inline component
interface SimpleDropZoneProps {
  onFilesAdd: (files: LocalQueryFile[]) => void;
}

function AddFilesButton({ onFilesAdd }: SimpleDropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(fileList: FileList) {
    const newFiles: LocalQueryFile[] = Array.from(fileList).map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      file,
    }));
    onFilesAdd(newFiles);
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files) {
            handleFiles(e.target.files);
            e.target.value = "";
          }
        }}
      />
      <Button
        variant="outlined"
        size="small"
        startIcon={<CloudUploadIcon />}
        onClick={() => inputRef.current?.click()}
      >
        Add Files
      </Button>
    </>
  );
}

interface QueryCardProps {
  query: EditableQuery;
  index: number;
  onUpdate: (query: EditableQuery) => void;
  onDelete: () => void;
  requireLabel: boolean;
}

function QueryCard({ query, index, onUpdate, onDelete, requireLabel }: QueryCardProps) {
  const [expanded, setExpanded] = useState(true);

  function handleFilesAdd(files: LocalQueryFile[]) {
    onUpdate({
      ...query,
      files: [...query.files, ...files],
    });
  }

  function handleFileDelete(fileId: string) {
    onUpdate({
      ...query,
      files: query.files.filter((f) => f.id !== fileId),
    });
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        overflow: "hidden",
        transition: "all 0.2s ease-in-out",
        "&:hover": {
          borderColor: "text.primary",
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 2,
          px: 2,
          py: 1.5,
          bgcolor: "action.hover",
          cursor: "pointer",
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Typography fontWeight={600}>Query {index + 1}</Typography>

        <Tag
          label={query.split}
          variant={query.split === "validation" ? "validation" : "test"}
          size="small"
        />

        {(query.files.length > 0 || (query.existingFiles?.length ?? 0) > 0) && (
          <Box display="flex" alignItems="center" gap={0.5} color="text.secondary">
            <FolderIcon fontSize="small" />
            <Typography variant="caption">
              {query.files.length + (query.existingFiles?.length ?? 0)} files
            </Typography>
          </Box>
        )}

        <Box flex={1} />

        <Tooltip title="Delete query">
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            sx={{ color: "error.main" }}
          >
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
      </Box>

      {/* Content */}
      <Collapse in={expanded}>
        <Box sx={{ p: 2 }}>
          <Stack spacing={3}>
            {/* Query Name */}
            <FormField label="Query Name">
              <TextField
                value={query.name}
                onChange={(e) => onUpdate({ ...query, name: e.target.value })}
                placeholder={`Query ${index + 1}`}
                fullWidth
                size="small"
              />
            </FormField>

            {/* Label */}
            <FormField
              label="Ground Truth Label"
              description={
                requireLabel
                  ? "Required for validation queries. This is the expected output."
                  : "Optional for test queries. Leave empty if unknown."
              }
              required={requireLabel}
            >
              <TextField
                value={query.label}
                onChange={(e) => onUpdate({ ...query, label: e.target.value })}
                placeholder={
                  requireLabel ? "Enter the expected output" : "Enter expected output (optional)"
                }
                fullWidth
                size="small"
                multiline
                minRows={2}
              />
            </FormField>

            {/* Files */}
            <Box>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Input Files
                </Typography>
                <AddFilesButton onFilesAdd={handleFilesAdd} />
              </Box>

              {/* Existing Files (read-only with preview) */}
              {query.existingFiles && query.existingFiles.length > 0 && (
                <Box mb={2}>
                  <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                    Existing files ({query.existingFiles.length})
                  </Typography>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                      gap: 2,
                    }}
                  >
                    {query.existingFiles.map((file) => (
                      <Paper
                        key={file.id}
                        variant="outlined"
                        sx={{
                          p: 1.5,
                          cursor: "pointer",
                          transition: "all 0.2s",
                          "&:hover": {
                            borderColor: "primary.main",
                            bgcolor: "action.hover",
                          },
                        }}
                      >
                        <Box
                          sx={{
                            height: 140,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            bgcolor: "action.hover",
                            borderRadius: 1,
                            mb: 1,
                            overflow: "hidden",
                          }}
                        >
                          <FilePreview
                            downloadUrl={file.download_url}
                            contentType={file.content_type}
                          />
                        </Box>
                        <Typography variant="body2" noWrap display="block" title={file.filename}>
                          {file.filename}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(file.size / 1024).toFixed(1)} KB
                        </Typography>
                      </Paper>
                    ))}
                  </Box>
                </Box>
              )}

              {/* New Files (removable with preview) */}
              {query.files.length > 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                    New files ({query.files.length})
                  </Typography>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                      gap: 2,
                    }}
                  >
                    {query.files.map((file) => (
                      <Paper
                        key={file.id}
                        variant="outlined"
                        sx={{
                          p: 1.5,
                          position: "relative",
                          transition: "all 0.2s",
                          "&:hover": {
                            borderColor: "primary.main",
                            "& .delete-btn": { opacity: 1 },
                          },
                        }}
                      >
                        <IconButton
                          className="delete-btn"
                          size="small"
                          onClick={() => handleFileDelete(file.id)}
                          sx={{
                            position: "absolute",
                            top: 4,
                            right: 4,
                            opacity: 0,
                            transition: "opacity 0.2s",
                            bgcolor: "background.paper",
                            "&:hover": { bgcolor: "error.light", color: "error.contrastText" },
                          }}
                        >
                          <DeleteOutlineIcon fontSize="small" />
                        </IconButton>
                        <Box
                          sx={{
                            height: 140,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            bgcolor: "action.hover",
                            borderRadius: 1,
                            mb: 1,
                            overflow: "hidden",
                          }}
                        >
                          <FilePreview file={file.file} />
                        </Box>
                        <Typography variant="body2" noWrap display="block" title={file.file.name}>
                          {file.file.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(file.file.size / 1024).toFixed(1)} KB
                        </Typography>
                      </Paper>
                    ))}
                  </Box>
                </Box>
              )}

              {/* Empty state */}
              {(!query.existingFiles || query.existingFiles.length === 0) && query.files.length === 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  <em>No files added</em>
                </Typography>
              )}
            </Box>
          </Stack>
        </Box>
      </Collapse>
    </Paper>
  );
}

interface DatasetSectionProps {
  title: string;
  description: string;
  type: "validation" | "test";
  queries: EditableQuery[];
  onAddQuery: () => void;
  onUpdateQuery: (query: EditableQuery) => void;
  onDeleteQuery: (queryId: number) => void;
  onZipUpload: (file: File) => void;
  requireLabel: boolean;
  recommendation?: string;
}

function DatasetSection({
  title,
  description,
  type,
  queries,
  onAddQuery,
  onUpdateQuery,
  onDeleteQuery,
  onZipUpload,
  requireLabel,
  recommendation,
}: DatasetSectionProps) {
  const filteredQueries = queries.filter((q) => q.split === type);
  const [dragOver, setDragOver] = useState(false);
  const zipInputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    const zipFile = files.find((f) => f.name.endsWith(".zip"));
    if (zipFile) {
      onZipUpload(zipFile);
    }
  }

  return (
    <Box>
      {/* Section Header */}
      <Box mb={2}>
        <Box display="flex" alignItems="center" gap={1} mb={0.5}>
          <Typography variant="h6" fontWeight={600}>
            {title}
          </Typography>
          <Tag
            label={`${filteredQueries.length}`}
            variant={type === "validation" ? "validation" : "test"}
            size="small"
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
        {recommendation && (
          <Alert severity="info" sx={{ mt: 1.5 }}>
            {recommendation}
          </Alert>
        )}
      </Box>

      {/* Query List */}
      <Stack spacing={2}>
        {filteredQueries.map((query, idx) => (
          <QueryCard
            key={query.id}
            query={query}
            index={idx}
            onUpdate={onUpdateQuery}
            onDelete={() => onDeleteQuery(query.id)}
            requireLabel={requireLabel}
          />
        ))}
      </Stack>

      {/* Empty State / Add Actions */}
      {filteredQueries.length === 0 ? (
        <Paper
          variant="outlined"
          sx={{
            p: 4,
            textAlign: "center",
            borderStyle: "dashed",
            bgcolor: dragOver ? "action.hover" : "transparent",
            transition: "all 0.2s",
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <Typography color="text.secondary" gutterBottom>
            No {type} queries yet
          </Typography>
          <Stack direction="row" spacing={2} justifyContent="center" mt={2}>
            <Button variant="outlined" startIcon={<AddIcon />} onClick={onAddQuery}>
              Add Query Manually
            </Button>
            <Button
              variant="outlined"
              startIcon={<UploadFileIcon />}
              onClick={() => zipInputRef.current?.click()}
            >
              Upload ZIP
            </Button>
            <input
              ref={zipInputRef}
              type="file"
              hidden
              accept=".zip"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onZipUpload(file);
                e.target.value = "";
              }}
            />
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block" mt={2}>
            ZIP structure: /query_name/files... with optional label.txt
          </Typography>
        </Paper>
      ) : (
        <Stack direction="row" spacing={2} mt={2}>
          <Button variant="outlined" size="small" startIcon={<AddIcon />} onClick={onAddQuery}>
            Add Query
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<UploadFileIcon />}
            onClick={() => zipInputRef.current?.click()}
          >
            Upload ZIP
          </Button>
          <input
            ref={zipInputRef}
            type="file"
            hidden
            accept=".zip"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onZipUpload(file);
              e.target.value = "";
            }}
          />
        </Stack>
      )}
    </Box>
  );
}

export default function DatasetStep({ queries, onQueriesChange }: DatasetStepProps) {
  // Add a new query of the specified type
  function addQuery(type: "validation" | "test") {
    const newQuery: EditableQuery = {
      id: queries.length,
      name: "",
      split: type,
      label: "",
      files: [],
    };
    onQueriesChange([...queries, newQuery]);
  }

  function updateQuery(updated: EditableQuery) {
    onQueriesChange(queries.map((q) => (q.id === updated.id ? updated : q)));
  }

  function deleteQuery(queryId: number) {
    const filtered = queries.filter((q) => q.id !== queryId);
    // Reindex
    onQueriesChange(filtered.map((q, idx) => ({ ...q, id: idx })));
  }

  async function handleZipUpload(file: File, type: "validation" | "test") {
    // TODO: Implement ZIP parsing
    console.log(`Uploading ${type} ZIP:`, file.name);
    // The actual implementation would parse the ZIP and create queries
  }

  // Count validation queries without labels (for warning)
  const validationWithoutLabels = queries.filter(
    (q) => q.split === "validation" && !q.label.trim()
  );

  return (
    <Stack spacing={4}>
      {/* Header */}
      <Box>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          Dataset Configuration
        </Typography>
        <Typography color="text.secondary">
          Upload your validation and test datasets. Each query contains input files and optionally
          a ground truth label.
        </Typography>
      </Box>

      {/* Warning for validation queries without labels */}
      {validationWithoutLabels.length > 0 && (
        <Alert severity="warning">
          {validationWithoutLabels.length} validation{" "}
          {validationWithoutLabels.length === 1 ? "query is" : "queries are"} missing labels.
        </Alert>
      )}

      {/* Validation Dataset Section */}
      <DatasetSection
        title="Validation Dataset"
        description="Used for model evaluation. Labels are required for each query."
        type="validation"
        queries={queries}
        onAddQuery={() => addQuery("validation")}
        onUpdateQuery={updateQuery}
        onDeleteQuery={deleteQuery}
        onZipUpload={(file) => handleZipUpload(file, "validation")}
        requireLabel={true}
        recommendation="Include at least 5 validation queries for better result performance."
      />

      <Divider />

      {/* Test Dataset Section */}
      <DatasetSection
        title="Test Dataset"
        description="Used for inference. Labels are optional."
        type="test"
        queries={queries}
        onAddQuery={() => addQuery("test")}
        onUpdateQuery={updateQuery}
        onDeleteQuery={deleteQuery}
        onZipUpload={(file) => handleZipUpload(file, "test")}
        requireLabel={false}
      />
    </Stack>
  );
}
