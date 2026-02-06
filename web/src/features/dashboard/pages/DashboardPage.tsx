/**
 * Dashboard (Home) Page.
 * Implements SRS 2.2 Dashboard requirements:
 * - Recent Activity Feed
 * - Quick Actions
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Stack,
  CircularProgress,
  Card,
  CardActionArea,
  Chip,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import AssignmentIcon from "@mui/icons-material/Assignment";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AccessTimeIcon from "@mui/icons-material/AccessTime";

import { ROUTES } from "../../../config/routes";
import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import { getMetricLabel } from "../../../config/metrics";

interface RecentItemProps {
  title: string;
  subtitle: string;
  metric?: string;
  date: string;
  onClick: () => void;
}

function RecentItem({ title, subtitle, metric, date, onClick }: RecentItemProps) {
  return (
    <Card variant="outlined" sx={{ mb: 1.5 }}>
      <CardActionArea onClick={onClick} sx={{ p: 2 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box flex={1} minWidth={0}>
            <Typography
              variant="subtitle1"
              fontWeight={500}
              noWrap
              sx={{ mb: 0.5 }}
            >
              {title}
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              noWrap
            >
              {subtitle}
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} alignItems="center" ml={2}>
            {metric && (
              <Chip size="small" label={getMetricLabel(metric)} variant="outlined" />
            )}
            <Typography variant="caption" color="text.secondary" whiteSpace="nowrap">
              {date}
            </Typography>
          </Stack>
        </Box>
      </CardActionArea>
    </Card>
  );
}

interface QuickActionProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
  variant?: "contained" | "outlined";
}

function QuickAction({ icon, title, description, onClick, variant = "outlined" }: QuickActionProps) {
  return (
    <Paper
      variant={variant === "contained" ? "elevation" : "outlined"}
      sx={{
        p: 3,
        cursor: "pointer",
        transition: "all 0.2s",
        bgcolor: variant === "contained" ? "primary.main" : "background.paper",
        color: variant === "contained" ? "primary.contrastText" : "text.primary",
        "&:hover": {
          transform: "translateY(-2px)",
          boxShadow: 3,
        },
      }}
      onClick={onClick}
    >
      <Box display="flex" alignItems="center" gap={2}>
        {icon}
        <Box>
          <Typography variant="subtitle1" fontWeight={600}>
            {title}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              opacity: variant === "contained" ? 0.9 : 0.7,
            }}
          >
            {description}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const [recentTasks, setRecentTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTasks()
      .then((tasks) => {
        // Sort by created_at descending and take top 5
        const sorted = [...tasks].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        setRecentTasks(sorted.slice(0, 5));
      })
      .finally(() => setLoading(false));
  }, []);

  function formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      {/* Welcome Section */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight={600} gutterBottom>
          Dashboard
        </Typography>
        <Typography color="text.secondary">
          Welcome to PRISM. Create and manage your ML task workflows.
        </Typography>
      </Box>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1fr 2fr" },
          gap: 4,
        }}
      >
        {/* Quick Actions Section */}
        <Box>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Quick Actions
          </Typography>
          <Stack spacing={2}>
            <QuickAction
              icon={<AddIcon fontSize="large" />}
              title="Create New Task"
              description="Define a new ML task with dataset"
              onClick={() => navigate(ROUTES.public.createTask)}
              variant="contained"
            />
            <QuickAction
              icon={<PlayArrowIcon fontSize="large" />}
              title="Run Workflow"
              description="Execute tasks with a pipeline"
              onClick={() => navigate(ROUTES.public.runTasks)}
            />
            <QuickAction
              icon={<AssignmentIcon fontSize="large" />}
              title="View All Tasks"
              description="Browse and manage your tasks"
              onClick={() => navigate(ROUTES.public.tasks)}
            />
          </Stack>
        </Box>

        {/* Recent Activity Section */}
        <Box>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" fontWeight={600}>
              Recent Tasks
            </Typography>
            {recentTasks.length > 0 && (
              <Button
                size="small"
                onClick={() => navigate(ROUTES.public.tasks)}
              >
                View All
              </Button>
            )}
          </Box>

          <Paper variant="outlined" sx={{ p: 2 }}>
            {loading ? (
              <Box display="flex" justifyContent="center" py={4}>
                <CircularProgress />
              </Box>
            ) : recentTasks.length === 0 ? (
              <Box textAlign="center" py={6}>
                <AccessTimeIcon sx={{ fontSize: 48, color: "text.secondary", mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No recent activity
                </Typography>
                <Typography color="text.secondary" mb={3}>
                  Get started by creating your first task
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => navigate(ROUTES.public.createTask)}
                >
                  Create Task
                </Button>
              </Box>
            ) : (
              <>
                {recentTasks.map((task) => (
                  <RecentItem
                    key={task.id}
                    title={task.name}
                    subtitle={task.description || "No description"}
                    metric={task.metric}
                    date={formatDate(task.created_at)}
                    onClick={() => navigate(`/tasks/${task.id}`)}
                  />
                ))}
              </>
            )}
          </Paper>

          {/* Stats Overview */}
          {recentTasks.length > 0 && (
            <Box mt={3}>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Overview
              </Typography>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(4, 1fr)" },
                  gap: 2,
                }}
              >
                <Paper variant="outlined" sx={{ p: 2, textAlign: "center" }}>
                  <Typography variant="h4" fontWeight={600} color="primary.main">
                    {recentTasks.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Recent Tasks
                  </Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2, textAlign: "center" }}>
                  <Typography variant="h4" fontWeight={600} color="primary.main">
                    {new Set(recentTasks.map((t) => t.metric)).size}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Metrics Used
                  </Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2, textAlign: "center" }}>
                  <Typography variant="h4" fontWeight={600} color="primary.main">
                    {recentTasks.reduce((acc, t) => acc + t.total_queries, 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Queries
                  </Typography>
                </Paper>
                <Paper variant="outlined" sx={{ p: 2, textAlign: "center" }}>
                  <Typography variant="h4" fontWeight={600} color="primary.main">
                    {recentTasks.filter((t) => t.validation_queries > 0).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    With Validation
                  </Typography>
                </Paper>
              </Box>
            </Box>
          )}
        </Box>
      </Box>
    </Container>
  );
}
