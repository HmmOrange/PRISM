/**
 * Task Library Page.
 * Implements SRS 2.3.3 - Task Library (All Tasks Page)
 * 
 * Features:
 * - Two-column layout
 * - Left Panel: Faceted Search & Filter
 * - Right Panel: Results with Grid/List View
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
  Chip,
  InputAdornment,
  Paper,
  Divider,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  ToggleButtonGroup,
  ToggleButton,
  Select,
  MenuItem,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import AddIcon from "@mui/icons-material/Add";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ViewListIcon from "@mui/icons-material/ViewList";
import GridViewIcon from "@mui/icons-material/GridView";
import ClearIcon from "@mui/icons-material/Clear";

import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import TaskCardGrid from "../components/TaskCardGrid";
import TaskListView from "../components/TaskListView";
import { AVAILABLE_METRICS } from "../../../config/metrics";
import { TASK_CATEGORIES } from "../../../config/taskTypes";
import { ROUTES } from "../../../config/routes";

type SortField = "created_at" | "name";
type SortOrder = "asc" | "desc";
type FilterLogic = "or" | "and" | "exact";
type ViewMode = "grid" | "list";

interface FilterState {
  searchQuery: string;
  selectedMetrics: string[];
  selectedTags: string[];
  filterLogic: FilterLogic;
  sortField: SortField;
  sortOrder: SortOrder;
}

const INITIAL_FILTERS: FilterState = {
  searchQuery: "",
  selectedMetrics: [],
  selectedTags: [],
  filterLogic: "or",
  sortField: "created_at",
  sortOrder: "desc",
};

export default function TaskLibraryPage() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [filters, setFilters] = useState<FilterState>(INITIAL_FILTERS);

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

  function updateFilter<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  function toggleMetricFilter(metric: string) {
    setFilters((prev) => ({
      ...prev,
      selectedMetrics: prev.selectedMetrics.includes(metric)
        ? prev.selectedMetrics.filter((m) => m !== metric)
        : [...prev.selectedMetrics, metric],
    }));
  }

  function toggleTagFilter(tag: string) {
    setFilters((prev) => ({
      ...prev,
      selectedTags: prev.selectedTags.includes(tag)
        ? prev.selectedTags.filter((t) => t !== tag)
        : [...prev.selectedTags, tag],
    }));
  }

  function clearFilters() {
    setFilters(INITIAL_FILTERS);
  }

  const hasActiveFilters = useMemo(() => {
    return (
      filters.searchQuery !== "" ||
      filters.selectedMetrics.length > 0 ||
      filters.selectedTags.length > 0
    );
  }, [filters]);

  const filteredAndSortedTasks = useMemo(() => {
    let result = [...tasks];

    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      result = result.filter(
        (task) =>
          task.name.toLowerCase().includes(query) ||
          task.description?.toLowerCase().includes(query)
      );
    }

    // Metric filter
    if (filters.selectedMetrics.length > 0) {
      result = result.filter((task) =>
        filters.selectedMetrics.includes(task.metric)
      );
    }

    // Tag filter (if we had tags on tasks, would apply here)
    // For now, tags are stored during creation but not on list items

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      if (filters.sortField === "created_at") {
        comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      } else if (filters.sortField === "name") {
        comparison = a.name.localeCompare(b.name);
      }
      return filters.sortOrder === "desc" ? -comparison : comparison;
    });

    return result;
  }, [tasks, filters]);

  // Get unique metrics from actual tasks
  const usedMetrics = useMemo(() => {
    const metrics = new Set(tasks.map((t) => t.metric));
    return AVAILABLE_METRICS.filter((m) => metrics.has(m.value));
  }, [tasks]);

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
          startIcon={<AddIcon />}
          onClick={() => navigate(ROUTES.public.createTask)}
        >
          New Task
        </Button>
      </Stack>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "250px 1fr" },
          gap: 3,
        }}
      >
        {/* Left Panel - Filters */}
        <Box>
          <Paper variant="outlined" sx={{ p: 2, position: "sticky", top: 16 }}>
            <Stack spacing={3}>
              {/* Header with Clear */}
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="h6" fontWeight={600}>
                  Filters
                </Typography>
                {hasActiveFilters && (
                  <Button size="small" onClick={clearFilters} startIcon={<ClearIcon />}>
                    Clear
                  </Button>
                )}
              </Box>

              <Divider />

              {/* Sort Options */}
              <Box>
                <FormControl fullWidth size="small">
                  <FormLabel sx={{ mb: 1, fontWeight: 500, fontSize: "0.875rem" }}>
                    Sort By
                  </FormLabel>
                  <Stack direction="row" spacing={1}>
                    <Select
                      value={filters.sortField}
                      onChange={(e) => updateFilter("sortField", e.target.value as SortField)}
                      size="small"
                      sx={{ flex: 1 }}
                    >
                      <MenuItem value="created_at">Date Created</MenuItem>
                      <MenuItem value="name">Name</MenuItem>
                    </Select>
                    <ToggleButtonGroup
                      value={filters.sortOrder}
                      exclusive
                      onChange={(_, value) => value && updateFilter("sortOrder", value)}
                      size="small"
                    >
                      <ToggleButton value="desc" title="Descending">
                        ↓
                      </ToggleButton>
                      <ToggleButton value="asc" title="Ascending">
                        ↑
                      </ToggleButton>
                    </ToggleButtonGroup>
                  </Stack>
                </FormControl>
              </Box>

              {/* Metric Filters */}
              {usedMetrics.length > 0 && (
                <Accordion defaultExpanded disableGutters elevation={0}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
                    <Typography fontWeight={500}>Metrics</Typography>
                    {filters.selectedMetrics.length > 0 && (
                      <Chip
                        size="small"
                        label={filters.selectedMetrics.length}
                        color="primary"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: 0 }}>
                    <Stack spacing={0.5}>
                      {usedMetrics.map((metric) => (
                        <Chip
                          key={metric.value}
                          label={metric.label}
                          size="small"
                          variant={
                            filters.selectedMetrics.includes(metric.value)
                              ? "filled"
                              : "outlined"
                          }
                          color={
                            filters.selectedMetrics.includes(metric.value)
                              ? "primary"
                              : "default"
                          }
                          onClick={() => toggleMetricFilter(metric.value)}
                          sx={{ cursor: "pointer", justifyContent: "flex-start" }}
                        />
                      ))}
                    </Stack>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Filter Logic */}
              {(filters.selectedMetrics.length > 1 || filters.selectedTags.length > 1) && (
                <Box>
                  <FormControl>
                    <FormLabel sx={{ mb: 1, fontWeight: 500, fontSize: "0.875rem" }}>
                      Filter Logic
                    </FormLabel>
                    <RadioGroup
                      value={filters.filterLogic}
                      onChange={(e) => updateFilter("filterLogic", e.target.value as FilterLogic)}
                      row
                    >
                      <FormControlLabel
                        value="or"
                        control={<Radio size="small" />}
                        label="Contains One"
                      />
                      <FormControlLabel
                        value="and"
                        control={<Radio size="small" />}
                        label="Contains All"
                      />
                      <FormControlLabel
                        value="exact"
                        control={<Radio size="small" />}
                        label="Exact Match"
                      />
                    </RadioGroup>
                  </FormControl>
                </Box>
              )}

              {/* Pipeline Categories */}
              <Accordion disableGutters elevation={0}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
                  <Typography fontWeight={500}>Categories</Typography>
                  {filters.selectedTags.length > 0 && (
                    <Chip
                      size="small"
                      label={filters.selectedTags.length}
                      color="primary"
                      sx={{ ml: 1 }}
                    />
                  )}
                </AccordionSummary>
                <AccordionDetails sx={{ px: 0, maxHeight: 300, overflow: "auto" }}>
                  <Stack spacing={1}>
                    {TASK_CATEGORIES.map((category) => (
                      <Chip
                        key={category.id}
                        label={category.name}
                        size="small"
                        variant={
                          filters.selectedTags.includes(category.id)
                            ? "filled"
                            : "outlined"
                        }
                        color={
                          filters.selectedTags.includes(category.id)
                            ? "primary"
                            : "default"
                        }
                        onClick={() => toggleTagFilter(category.id)}
                        sx={{ cursor: "pointer" }}
                      />
                    ))}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Stack>
          </Paper>
        </Box>

        {/* Right Panel - Results */}
        <Box>
          {/* Search and View Toggle */}
          <Stack direction="row" spacing={2} mb={3} alignItems="center">
            <TextField
              placeholder="Search tasks..."
              size="small"
              value={filters.searchQuery}
              onChange={(e) => updateFilter("searchQuery", e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon color="action" />
                  </InputAdornment>
                ),
              }}
              sx={{ flex: 1, maxWidth: 400 }}
            />

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
                  startIcon={<AddIcon />}
                  onClick={() => navigate(ROUTES.public.createTask)}
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
        </Box>
      </Box>
    </Container>
  );
}
