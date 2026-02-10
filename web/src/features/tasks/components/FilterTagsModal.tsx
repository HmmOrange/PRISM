/**
 * Filter & Sort Modal Component.
 * Modal for sorting and filtering tasks by categories or pipeline tags.
 */

import { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton,
  Typography,
  Box,
  Stack,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  ToggleButtonGroup,
  ToggleButton,
  Divider,
  TextField,
  InputAdornment,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SearchIcon from "@mui/icons-material/Search";

import { Tag } from "../../../components";
import { TASK_CATEGORIES } from "../../../config/taskTypes";

export type SortField = "none" | "created_at" | "name";
export type SortOrder = "asc" | "desc";
export type FilterType = "none" | "category" | "tags";

export interface FilterState {
  sortField: SortField;
  sortOrder: SortOrder;
  filterType: FilterType;
  selectedCategories: string[];
  selectedTags: string[];
}

interface FilterTagsModalProps {
  open: boolean;
  onClose: () => void;
  filterState: FilterState;
  onApply: (state: FilterState) => void;
}

export const INITIAL_FILTER_STATE: FilterState = {
  sortField: "none",
  sortOrder: "desc",
  filterType: "none",
  selectedCategories: [],
  selectedTags: [],
};

export default function FilterTagsModal({
  open,
  onClose,
  filterState,
  onApply,
}: FilterTagsModalProps) {
  const [tempState, setTempState] = useState<FilterState>(filterState);
  const [tagSearch, setTagSearch] = useState("");

  // Reset temp state when modal opens
  function handleOpen() {
    setTempState(filterState);
    setTagSearch("");
  }

  function updateTempState<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    setTempState((prev) => ({ ...prev, [key]: value }));
  }

  function toggleCategory(categoryId: string) {
    setTempState((prev) => ({
      ...prev,
      selectedCategories: prev.selectedCategories.includes(categoryId)
        ? prev.selectedCategories.filter((c) => c !== categoryId)
        : [...prev.selectedCategories, categoryId],
    }));
  }

  function toggleTag(tag: string) {
    setTempState((prev) => ({
      ...prev,
      selectedTags: prev.selectedTags.includes(tag)
        ? prev.selectedTags.filter((t) => t !== tag)
        : [...prev.selectedTags, tag],
    }));
  }

  function handleApply() {
    onApply(tempState);
    onClose();
  }

  function handleClear() {
    setTempState(INITIAL_FILTER_STATE);
  }

  // Normalize string for search
  function normalizeForSearch(str: string): string {
    return str.toLowerCase().replace(/-/g, "");
  }

  // Filter categories and tasks based on search
  const filteredCategories = TASK_CATEGORIES.map((category) => ({
    ...category,
    tasks: category.tasks.filter((task) =>
      normalizeForSearch(task).includes(normalizeForSearch(tagSearch))
    ),
  })).filter((category) => category.tasks.length > 0);

  const hasActiveFilters =
    tempState.sortField !== "none" ||
    tempState.filterType !== "none" ||
    tempState.selectedCategories.length > 0 ||
    tempState.selectedTags.length > 0;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      TransitionProps={{ onEnter: handleOpen }}
      PaperProps={{
        sx: {
          borderRadius: 2,
          minHeight: "70vh",
          maxHeight: "90vh",
        },
      }}
    >
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          pb: 1,
        }}
      >
        <Typography variant="h6" fontWeight={600}>
          Sort & Filter
        </Typography>
        <IconButton
          size="small"
          onClick={onClose}
          sx={{ color: "text.secondary" }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers sx={{ py: 3 }}>
        <Stack spacing={3}>
          {/* Sort Section */}
          <Box>
            <FormControl fullWidth>
              <FormLabel sx={{ mb: 1, fontWeight: 600 }}>Sort By</FormLabel>
              <Stack direction="row" spacing={2} alignItems="center">
                <RadioGroup
                  row
                  value={tempState.sortField}
                  onChange={(e) => updateTempState("sortField", e.target.value as SortField)}
                >
                  <FormControlLabel value="none" control={<Radio size="small" color="secondary" />} label="None" />
                  <FormControlLabel value="created_at" control={<Radio size="small" color="secondary" />} label="Date Created" />
                  <FormControlLabel value="name" control={<Radio size="small" color="secondary" />} label="Name" />
                </RadioGroup>

                {tempState.sortField !== "none" && (
                  <ToggleButtonGroup
                    value={tempState.sortOrder}
                    exclusive
                    onChange={(_, value) => value && updateTempState("sortOrder", value)}
                    size="small"
                    color="secondary"
                  >
                    <ToggleButton value="asc">Asc</ToggleButton>
                    <ToggleButton value="desc">Desc</ToggleButton>
                  </ToggleButtonGroup>
                )}
              </Stack>
            </FormControl>
          </Box>

          <Divider />

          {/* Filter Type Selection */}
          <Box>
            <FormControl fullWidth>
              <FormLabel sx={{ mb: 1, fontWeight: 600 }}>Filter By</FormLabel>
              <RadioGroup
                row
                value={tempState.filterType}
                onChange={(e) => {
                  const newType = e.target.value as FilterType;
                  updateTempState("filterType", newType);
                  // Clear selections when switching filter type
                  if (newType === "none") {
                    setTempState((prev) => ({
                      ...prev,
                      filterType: newType,
                      selectedCategories: [],
                      selectedTags: [],
                    }));
                  }
                }}
              >
                <FormControlLabel value="none" control={<Radio size="small" color="secondary" />} label="None" />
                <FormControlLabel value="category" control={<Radio size="small" color="secondary" />} label="Category" />
                <FormControlLabel value="tags" control={<Radio size="small" color="secondary" />} label="Pipeline Tags" />
              </RadioGroup>
            </FormControl>
          </Box>

          {/* Category Selection */}
          {tempState.filterType === "category" && (
            <Box>
              <Typography variant="body2" color="text.secondary" mb={1.5}>
                Select categories to filter by (tasks with any selected category will be shown)
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                {TASK_CATEGORIES.map((category) => (
                  <Tag
                    key={category.id}
                    label={category.name}
                    variant={
                      tempState.selectedCategories.includes(category.id)
                        ? "selected"
                        : "default"
                    }
                    onClick={() => toggleCategory(category.id)}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Pipeline Tags Selection */}
          {tempState.filterType === "tags" && (
            <Box>
              <Typography variant="body2" color="text.secondary" mb={1.5}>
                Select pipeline tags to filter by (tasks with any selected tag will be shown)
              </Typography>

              <TextField
                placeholder="Search tags..."
                size="small"
                fullWidth
                value={tagSearch}
                onChange={(e) => setTagSearch(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon fontSize="small" color="action" />
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <Box sx={{ maxHeight: 400, overflow: "auto" }}>
                <Stack spacing={2.5}>
                  {filteredCategories.map((category) => (
                    <Box key={category.id}>
                      {/* Category Header */}
                      <Typography
                        variant="subtitle2"
                        sx={{
                          color: "text.secondary",
                          fontWeight: 600,
                          mb: 1,
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                          fontSize: "0.75rem",
                        }}
                      >
                        {category.name}
                      </Typography>

                      {/* Task Tags */}
                      <Box display="flex" flexWrap="wrap" gap={0.75}>
                        {category.tasks.map((task) => (
                          <Tag
                            key={task}
                            label={task}
                            size="small"
                            variant={
                              tempState.selectedTags.includes(task)
                                ? "selected"
                                : "default"
                            }
                            onClick={() => toggleTag(task)}
                          />
                        ))}
                      </Box>
                    </Box>
                  ))}
                </Stack>

                {filteredCategories.length === 0 && tagSearch && (
                  <Typography color="text.secondary" variant="body2" textAlign="center" py={2}>
                    No tags match "{tagSearch}"
                  </Typography>
                )}
              </Box>
            </Box>
          )}
        </Stack>
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2, justifyContent: "space-between" }}>
        <Button
          onClick={handleClear}
          color="inherit"
          disabled={!hasActiveFilters}
        >
          Clear All
        </Button>
        <Stack direction="row" spacing={1}>
          <Button onClick={onClose} color="inherit">
            Cancel
          </Button>
          <Button onClick={handleApply} variant="contained">
            Apply
          </Button>
        </Stack>
      </DialogActions>
    </Dialog>
  );
}
