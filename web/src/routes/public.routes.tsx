/**
 * Public routes configuration.
 * These routes are accessible without authentication.
 * Implements SRS 2.1.2 and 2.2 Navigation requirements.
 */

import type { RouteObject } from "react-router";
import { Navigate } from "react-router-dom";
import { ROUTES } from "../config/routes";

// Layouts
import MainLayout from "../components/layout/MainLayout";

// Auth pages (with guest guard)
import LoginPage from "../features/auth/pages/LoginPage";
import RegisterPage from "../features/auth/pages/RegisterPage";
import { GuestGuard } from "../features/auth";

// Dashboard
import { DashboardPage } from "../features/dashboard";

// Task pages
import TaskLibraryPage from "../features/tasks/pages/TaskLibraryPage";
import CreateTaskWizardPage from "../features/tasks/pages/CreateTaskWizardPage";
import TaskDetailPage from "../features/tasks/pages/TaskDetailPage";

// Workflow pages
import RunPage from "../features/workflows/pages/RunPage";

export const publicRoutes: RouteObject[] = [
  // Auth routes (guest only)
  {
    path: ROUTES.public.login,
    element: (
      <GuestGuard>
        <LoginPage />
      </GuestGuard>
    ),
  },
  {
    path: ROUTES.public.register,
    element: (
      <GuestGuard>
        <RegisterPage />
      </GuestGuard>
    ),
  },

  // Main app routes
  {
    element: <MainLayout />,
    children: [
      // Home redirects to dashboard
      {
        path: ROUTES.public.home,
        element: <Navigate to={ROUTES.public.dashboard} replace />,
      },
      
      // Dashboard
      {
        path: ROUTES.public.dashboard,
        element: <DashboardPage />,
      },

      // Task routes
      {
        path: ROUTES.public.tasks,
        element: <TaskLibraryPage />,
      },
      {
        path: ROUTES.public.createTask,
        element: <CreateTaskWizardPage />,
      },
      {
        path: ROUTES.public.taskDetail,
        element: <TaskDetailPage />,
      },

      // Workflow/Run routes
      {
        path: ROUTES.public.runTasks,
        element: <RunPage />,
      },
    ],
  },
];
