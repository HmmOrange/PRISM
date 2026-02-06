/**
 * Authenticated routes configuration.
 * These routes require authentication.
 * Implements SRS 2.1.1 Guard Clauses.
 */

import type { RouteObject } from "react-router";

// Note: These imports will be used when authentication is fully implemented
// import { Navigate } from "react-router-dom";
// import { ROUTES } from "../config/routes";
// import MainLayout from "../components/layout/MainLayout";
// import { AuthGuard } from "../features/auth";
// import { DashboardPage } from "../features/dashboard";
// import TaskLibraryPage from "../features/tasks/pages/TaskLibraryPage";
// import CreateTaskWizardPage from "../features/tasks/pages/CreateTaskWizardPage";
// import TaskDetailPage from "../features/tasks/pages/TaskDetailPage";
// import RunPage from "../features/workflows/pages/RunPage";

export const authedRoutes: RouteObject[] = [
  // All authenticated routes wrapped with AuthGuard
  // Currently disabled - using public routes for development
  // Uncomment when authentication is fully implemented
  
  // {
  //   element: <ProtectedLayout />,
  //   children: [
  //     {
  //       path: ROUTES.authed.dashboard,
  //       element: <DashboardPage />,
  //     },
  //     {
  //       path: ROUTES.authed.tasks,
  //       element: <TaskLibraryPage />,
  //     },
  //     {
  //       path: ROUTES.authed.createTask,
  //       element: <CreateTaskWizardPage />,
  //     },
  //     {
  //       path: ROUTES.authed.taskDetail,
  //       element: <TaskDetailPage />,
  //     },
  //     {
  //       path: ROUTES.authed.run,
  //       element: <RunPage />,
  //     },
  //   ],
  // },
];
