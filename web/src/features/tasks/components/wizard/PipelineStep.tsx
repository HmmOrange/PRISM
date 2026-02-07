/**
 * Stage 2: Pipeline Configuration.
 * Implements SRS 2.3.1 Stage 2:
 * - Flat list of pipeline tags (not mandatory)
 * - Grouped by category for organization
 * - Uses universal Tag component
 */

import { useState } from "react";
import {
  Stack,
  Typography,
  Box,
  TextField,
  InputAdornment,
  Paper,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";

import { Tag } from "../../../../components";
import { TASK_CATEGORIES } from "../../../../config/taskTypes";

interface PipelineStepProps {
  selectedTags: string[];
  error?: string;
  onToggleTag: (tag: string) => void;
}

export default function PipelineStep({
  selectedTags,
  onToggleTag,
}: PipelineStepProps) {
  const [searchQuery, setSearchQuery] = useState("");

  // Normalize string for search - remove dashes and convert to lowercase
  function normalizeForSearch(str: string): string {
    return str.toLowerCase().replace(/-/g, "");
  }

  // Filter categories and tasks based on search
  const filteredCategories = TASK_CATEGORIES.map((category) => ({
    ...category,
    tasks: category.tasks.filter((task) =>
      normalizeForSearch(task).includes(normalizeForSearch(searchQuery))
    ),
  })).filter((category) => category.tasks.length > 0);

  return (
    <Stack spacing={4}>
      {/* Header */}
      <Box>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          Pipeline Tags
          {selectedTags.length > 0 && (
            <Typography
              component="span"
              sx={{
                ml: 1,
                fontSize: "1rem",
                fontWeight: 400,
                color: "text.secondary",
              }}
            >
              ({selectedTags.length} selected)
            </Typography>
          )}
        </Typography>
        <Typography color="text.secondary">
          Optionally tag your task with relevant ML task types. This helps organize and categorize tasks.
        </Typography>
      </Box>

      {/* Search */}
      <TextField
        placeholder="Search task types..."
        size="small"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon color="action" />
            </InputAdornment>
          ),
        }}
        sx={{ maxWidth: 400 }}
      />

      {/* Category Sections */}
      <Stack spacing={3}>
        {filteredCategories.map((category) => (
          <Box key={category.id}>
            {/* Category Header */}
            <Typography
              variant="subtitle2"
              sx={{
                color: "text.secondary",
                fontWeight: 600,
                mb: 1.5,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                fontSize: "0.75rem",
              }}
            >
              {category.name}
            </Typography>

            {/* Task Tags */}
            <Box display="flex" flexWrap="wrap" gap={1}>
              {category.tasks.map((task) => (
                <Tag
                  key={task}
                  label={task}
                  variant={selectedTags.includes(task) ? "selected" : "default"}
                  onClick={() => onToggleTag(task)}
                />
              ))}
            </Box>
          </Box>
        ))}
      </Stack>

      {/* Empty State */}
      {filteredCategories.length === 0 && searchQuery && (
        <Paper variant="outlined" sx={{ p: 4, textAlign: "center" }}>
          <Typography color="text.secondary">
            No task types match "{searchQuery}"
          </Typography>
        </Paper>
      )}
    </Stack>
  );
}
