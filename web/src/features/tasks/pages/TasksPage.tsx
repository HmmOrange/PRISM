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
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import AddIcon from "@mui/icons-material/Add";
import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";

import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import TaskCardGrid from "../components/TaskCardGrid";
import { AVAILABLE_METRICS } from "../../../config/metrics";
import { ROUTES } from "../../../config/routes";

export default function TasksPage() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);

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

  function toggleMetricFilter(metric: string) {
    setSelectedMetrics((prev) =>
      prev.includes(metric)
        ? prev.filter((m) => m !== metric)
        : [...prev, metric]
    );
  }

  const filteredTasks = useMemo(() => {
    return tasks.filter((task) => {
      // Search filter
      const matchesSearch =
        !searchQuery ||
        task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        task.description?.toLowerCase().includes(searchQuery.toLowerCase());

      // Metric filter
      const matchesMetric =
        selectedMetrics.length === 0 || selectedMetrics.includes(task.metric);

      return matchesSearch && matchesMetric;
    });
  }, [tasks, searchQuery, selectedMetrics]);

  // Get unique metrics from actual tasks
  const usedMetrics = useMemo(() => {
    const metrics = new Set(tasks.map((t) => t.metric));
    return AVAILABLE_METRICS.filter((m) => metrics.has(m.value));
  }, [tasks]);

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        justifyContent="space-between"
        alignItems={{ xs: "stretch", sm: "center" }}
        spacing={2}
        mb={3}
      >
        <Typography variant="h4">Tasks</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => navigate(ROUTES.public.createTask)}
        >
          New Task
        </Button>
      </Stack>

      {/* Search and filters */}
      <Stack spacing={2} mb={3}>
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
          sx={{ maxWidth: 400 }}
        />

        {usedMetrics.length > 0 && (
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center", mr: 1 }}>
              Filter by metric:
            </Typography>
            {usedMetrics.map((metric) => (
              <Chip
                key={metric.value}
                label={metric.label}
                size="small"
                variant={selectedMetrics.includes(metric.value) ? "filled" : "outlined"}
                color={selectedMetrics.includes(metric.value) ? "primary" : "default"}
                onClick={() => toggleMetricFilter(metric.value)}
              />
            ))}
            {selectedMetrics.length > 0 && (
              <Chip
                label="Clear"
                size="small"
                variant="outlined"
                onDelete={() => setSelectedMetrics([])}
              />
            )}
          </Stack>
        )}
      </Stack>

      {loading ? (
        <Box display="flex" justifyContent="center" mt={6}>
          <CircularProgress />
        </Box>
      ) : filteredTasks.length === 0 ? (
        <Box
          sx={{
            textAlign: "center",
            py: 8,
            color: "text.secondary",
          }}
        >
          <Typography variant="h6" gutterBottom>
            {tasks.length === 0 ? "No tasks yet" : "No tasks match your filters"}
          </Typography>
          <Typography variant="body2" mb={2}>
            {tasks.length === 0
              ? "Create your first task to get started"
              : "Try adjusting your search or filter criteria"}
          </Typography>
          {tasks.length === 0 && (
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={() => navigate(ROUTES.public.createTask)}
            >
              Create Task
            </Button>
          )}
        </Box>
      ) : (
        <TaskCardGrid
          tasks={filteredTasks}
          onTaskDeleted={handleTaskDeleted}
        />
      )}
    </Container>
  );
}
