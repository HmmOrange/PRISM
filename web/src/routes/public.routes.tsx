import type { RouteObject } from "react-router";
import { ROUTES } from "../config/routes";
import MainLayout from "../components/layout/MainLayout";
import TasksPage from "../features/tasks/pages/TasksPage";
import CreateTaskPage from "../features/tasks/pages/CreateTaskPage";

export const publicRoutes: RouteObject[] = [
  {
    element: <MainLayout />,
    children: [
      {
        path: ROUTES.public.home,
        element: <TasksPage />,
      },
      {
        path: ROUTES.public.createTask,
        element: <CreateTaskPage />,
      },
    ],
  },
];
