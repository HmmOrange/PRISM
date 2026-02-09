/**
 * Authenticated routes configuration.
 * These routes require authentication.
 * Implements SRS 2.1.1 Guard Clauses.
 */

import type { RouteObject } from "react-router";
import { Navigate } from "react-router-dom";
import { ROUTES } from "../config/routes";

// Layouts & Guards
import MainLayout from "../components/layout/MainLayout";
import { AuthGuard } from "../features/auth";

// Dashboard
import { DashboardPage } from "../features/dashboard";

// Task pages
import TaskLibraryPage from "../features/tasks/pages/TaskLibraryPage";
import CreateTaskWizardPage from "../features/tasks/pages/CreateTaskWizardPage";
import TaskDetailPage from "../features/tasks/pages/TaskDetailPage";

// Workflow pages
import RunPage from "../features/workflows/pages/RunPage";

export const authedRoutes: RouteObject[] = [
  {
    element: (
      <AuthGuard>
        <MainLayout />
      </AuthGuard>
    ),
    children: [
      // Home redirects to dashboard
      {
        path: ROUTES.authed.dashboard,
        element: <DashboardPage />,
      },
      {
        path: "/",
        element: <Navigate to={ROUTES.authed.dashboard} replace />,
      },

      // Task routes
      {
        path: ROUTES.authed.tasks,
        element: <TaskLibraryPage />,
      },
      {
        path: ROUTES.authed.createTask,
        element: <CreateTaskWizardPage />,
      },
      {
        path: ROUTES.authed.taskDetail,
        element: <TaskDetailPage />,
      },

      // Workflow/Run routes
      {
        path: ROUTES.authed.run,
        element: <RunPage />,
      },
    ],
  },
];
