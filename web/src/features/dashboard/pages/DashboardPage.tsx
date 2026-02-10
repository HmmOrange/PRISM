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
  CircularProgress,
  Grid,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import AssignmentIcon from "@mui/icons-material/Assignment";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AccessTimeIcon from "@mui/icons-material/AccessTime";

import { ROUTES } from "../../../config/routes";
import { getTasks } from "../../../api/tasks.api";
import type { TaskListItem } from "../../../types/tasks.types";
import { useAuth } from "../../auth";
import TaskCard from "../../tasks/components/TaskCard";

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
  const { user } = useAuth();
  const [recentTasks, setRecentTasks] = useState<TaskListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTasks()
      .then((tasks) => {
        // Sort by created_at descending and take top 6
        const sorted = [...tasks].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        setRecentTasks(sorted.slice(0, 6));
      })
      .finally(() => setLoading(false));
  }, []);

  function handleTaskDeleted(taskId: string) {
    setRecentTasks((prev) => prev.filter((t) => t.id !== taskId));
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
      {/* Welcome Section */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight={600} gutterBottom>
          Hello, {user?.username || "User"}
        </Typography>
        <Typography color="text.secondary">
          Welcome to PRISM. Create and manage your ML task workflows.
        </Typography>
      </Box>

      {/* Quick Actions Section - Now at the top */}
      <Box mb={4}>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Quick Actions
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", sm: "repeat(3, 1fr)" },
            gap: 2,
          }}
        >
          <QuickAction
            icon={<AddIcon fontSize="large" />}
            title="Create New Task"
            description="Define a new ML task with dataset"
            onClick={() => navigate(ROUTES.authed.createTask)}
            variant="contained"
          />
          <QuickAction
            icon={<PlayArrowIcon fontSize="large" />}
            title="Run Workflow"
            description="Execute tasks with a pipeline"
            onClick={() => navigate(ROUTES.authed.run)}
          />
          <QuickAction
            icon={<AssignmentIcon fontSize="large" />}
            title="View All Tasks"
            description="Browse and manage your tasks"
            onClick={() => navigate(ROUTES.authed.tasks)}
          />
        </Box>
      </Box>

      {/* Recent Tasks Section */}
      <Box>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight={600}>
            Recent Tasks
          </Typography>
          {recentTasks.length > 0 && (
            <Button
              size="small"
              onClick={() => navigate(ROUTES.authed.tasks)}
            >
              View All
            </Button>
          )}
        </Box>

        {loading ? (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        ) : recentTasks.length === 0 ? (
          <Paper variant="outlined" sx={{ p: 2 }}>
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
                color="secondary"
                startIcon={<AddIcon />}
                onClick={() => navigate(ROUTES.authed.createTask)}
              >
                Create Task
              </Button>
            </Box>
          </Paper>
        ) : (
          <Grid container spacing={2}>
            {recentTasks.map((task) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={task.id}>
                <TaskCard task={task} onDeleted={handleTaskDeleted} />
              </Grid>
            ))}
          </Grid>
        )}
      </Box>
    </Container>
  );
}
