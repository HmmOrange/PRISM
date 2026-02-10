/**
 * Task Library Page.
 * Implements SRS 2.3.3 - Task Library (All Tasks Page)
 * 
 * Features:
 * - Search bar with Sort & Filter button
 * - Grid/List view toggle
 * - Filter modal for sorting and filtering
 */

import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Typography,
  CircularProgress,
  Box,
  TextField,
  Button,
  Stack,
  InputAdornment,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import AddIcon from "@mui/icons-material/Add";
import ViewListIcon from "@mui/icons-material/ViewList";
import GridViewIcon from "@mui/icons-material/GridView";
import FilterListIcon from "@mui/icons-material/FilterList";

import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import TaskCardGrid from "../components/TaskCardGrid";
import TaskListView from "../components/TaskListView";
import FilterTagsModal, { INITIAL_FILTER_STATE } from "../components/FilterTagsModal";
import type { FilterState } from "../components/FilterTagsModal";
import { Tag } from "../../../components";
import { TASK_CATEGORIES } from "../../../config/taskTypes";
import { ROUTES } from "../../../config/routes";

type ViewMode = "grid" | "list";

export default function TaskLibraryPage() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [filterState, setFilterState] = useState<FilterState>(INITIAL_FILTER_STATE);
  const [filterModalOpen, setFilterModalOpen] = useState(false);

  useEffect(() => {
    reloadTasks();
  }, []);

  function reloadTasks() {
    setLoading(true);
    getTasks()
      .then(setTasks)
      .finally(() => setLoading(false));
  }

  function handleTaskDeleted(taskId: string) {
    setTasks((prev) => prev.filter((t) => t.id !== taskId));
  }

  function handleApplyFilter(newState: FilterState) {
    setFilterState(newState);
  }

  function clearFilters() {
    setFilterState(INITIAL_FILTER_STATE);
  }

  // Check if any filter is active
  const hasActiveFilters =
    filterState.sortField !== "none" ||
    filterState.filterType !== "none" ||
    filterState.selectedCategories.length > 0 ||
    filterState.selectedTags.length > 0;

  // Get tags for a category
  function getCategoryTags(categoryId: string): string[] {
    const category = TASK_CATEGORIES.find((c) => c.id === categoryId);
    return category ? category.tasks : [];
  }

  const filteredAndSortedTasks = useMemo(() => {
    let result = [...tasks];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (task) =>
          task.name.toLowerCase().includes(query) ||
          task.description?.toLowerCase().includes(query)
      );
    }

    // Category filter (has any tag from selected categories)
    if (filterState.filterType === "category" && filterState.selectedCategories.length > 0) {
      const categoryTags = filterState.selectedCategories.flatMap(getCategoryTags);
      result = result.filter(
        (task) =>
          task.pipeline_tags &&
          categoryTags.some((tag) => task.pipeline_tags?.includes(tag))
      );
    }

    // Pipeline tag filter
    if (filterState.filterType === "tags" && filterState.selectedTags.length > 0) {
      result = result.filter(
        (task) =>
          task.pipeline_tags &&
          filterState.selectedTags.some((tag) => task.pipeline_tags?.includes(tag))
      );
    }

    // Sorting
    if (filterState.sortField !== "none") {
      result.sort((a, b) => {
        let comparison = 0;
        if (filterState.sortField === "name") {
          comparison = a.name.localeCompare(b.name);
        } else if (filterState.sortField === "created_at") {
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        }
        return filterState.sortOrder === "desc" ? -comparison : comparison;
      });
    }

    return result;
  }, [tasks, searchQuery, filterState]);

  // Get display label for active filters
  function getFilterLabel(): string {
    const parts: string[] = [];
    
    if (filterState.sortField !== "none") {
      const sortLabel = filterState.sortField === "created_at" ? "Date" : "Name";
      parts.push(`Sort: ${sortLabel}`);
    }
    
    if (filterState.filterType === "category" && filterState.selectedCategories.length > 0) {
      parts.push(`${filterState.selectedCategories.length} categories`);
    }
    
    if (filterState.filterType === "tags" && filterState.selectedTags.length > 0) {
      parts.push(`${filterState.selectedTags.length} tags`);
    }

    return parts.length > 0 ? parts.join(", ") : "";
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      {/* Header */}
      <Stack
        direction={{ xs: "column", sm: "row" }}
        justifyContent="space-between"
        alignItems={{ xs: "stretch", sm: "center" }}
        spacing={2}
        mb={3}
      >
        <Typography variant="h4">Task Library</Typography>
        <Button
          variant="contained"
          color="secondary"
          startIcon={<AddIcon />}
          onClick={() => navigate(ROUTES.authed.createTask)}
        >
          New Task
        </Button>
      </Stack>

      {/* Search, Filter, and View Toggle */}
      <Stack spacing={2} mb={3}>
        <Stack direction="row" spacing={2} alignItems="center">
          <TextField
            placeholder="Search tasks..."
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
            sx={{ flex: 1, maxWidth: 400 }}
          />

          <Button
            variant={hasActiveFilters ? "contained" : "outlined"}
            color={hasActiveFilters ? "secondary" : "primary"}
            startIcon={<FilterListIcon />}
            onClick={() => setFilterModalOpen(true)}
          >
            {hasActiveFilters ? getFilterLabel() : "Sort & Filter"}
          </Button>

          {hasActiveFilters && (
            <Button variant="text" size="small" onClick={clearFilters}>
              Clear
            </Button>
          )}

          <Box flex={1} />

          <Typography variant="body2" color="text.secondary">
            {filteredAndSortedTasks.length} task{filteredAndSortedTasks.length !== 1 && "s"}
          </Typography>

          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, value) => value && setViewMode(value)}
            size="small"
          >
            <ToggleButton value="grid" title="Grid View">
              <GridViewIcon />
            </ToggleButton>
            <ToggleButton value="list" title="List View">
              <ViewListIcon />
            </ToggleButton>
          </ToggleButtonGroup>
        </Stack>

        {/* Active filter tags display */}
        {filterState.filterType === "tags" && filterState.selectedTags.length > 0 && (
          <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap alignItems="center">
            <Typography variant="body2" color="text.secondary" mr={0.5}>
              Filtering by:
            </Typography>
            {filterState.selectedTags.slice(0, 5).map((tag) => (
              <Tag key={tag} label={tag} size="small" variant="selected" />
            ))}
            {filterState.selectedTags.length > 5 && (
              <Tag
                label={`+${filterState.selectedTags.length - 5} more`}
                size="small"
                variant="default"
              />
            )}
          </Stack>
        )}

        {filterState.filterType === "category" && filterState.selectedCategories.length > 0 && (
          <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap alignItems="center">
            <Typography variant="body2" color="text.secondary" mr={0.5}>
              Filtering by categories:
            </Typography>
            {filterState.selectedCategories.map((catId) => {
              const cat = TASK_CATEGORIES.find((c) => c.id === catId);
              return cat ? (
                <Tag key={catId} label={cat.name} size="small" variant="selected" />
              ) : null;
            })}
          </Stack>
        )}
      </Stack>

      {/* Filter Modal */}
      <FilterTagsModal
        open={filterModalOpen}
        onClose={() => setFilterModalOpen(false)}
        filterState={filterState}
        onApply={handleApplyFilter}
      />

      {/* Results */}
      {loading ? (
        <Box display="flex" justifyContent="center" mt={6}>
          <CircularProgress />
        </Box>
      ) : filteredAndSortedTasks.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 6, textAlign: "center" }}>
          <Typography variant="h6" gutterBottom color="text.secondary">
            {tasks.length === 0 ? "No tasks yet" : "No tasks match your filters"}
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            {tasks.length === 0
              ? "Create your first task to get started"
              : "Try adjusting your search or filter criteria"}
          </Typography>
          {tasks.length === 0 ? (
            <Button
              variant="contained"
              color="secondary"
              startIcon={<AddIcon />}
              onClick={() => navigate(ROUTES.authed.createTask)}
            >
              Create Task
            </Button>
          ) : (
            <Button variant="outlined" onClick={clearFilters}>
              Clear Filters
            </Button>
          )}
        </Paper>
      ) : viewMode === "grid" ? (
        <TaskCardGrid tasks={filteredAndSortedTasks} onTaskDeleted={handleTaskDeleted} />
      ) : (
        <TaskListView tasks={filteredAndSortedTasks} onTaskDeleted={handleTaskDeleted} />
      )}
    </Container>
  );
}
