/**
 * Application routes configuration.
 * Implements SRS 2.2 Navigation structure.
 */
export const ROUTES = {
  public: {
    // Auth pages (guest only)
    login: "/login",
    register: "/register",
    
    // Landing/Home
    home: "/",
    dashboard: "/dashboard",
    
    // Task Management
    tasks: "/tasks",
    createTask: "/tasks/new",
    taskDetail: "/tasks/:taskId",
    
    // Workflow/Run
    runTasks: "/run",
    projects: "/projects",
  },
  authed: {
    // Protected routes (require authentication)
    dashboard: "/dashboard",
    tasks: "/tasks",
    createTask: "/tasks/new",
    taskDetail: "/tasks/:taskId",
    run: "/run",
    projects: "/projects",
    profile: "/profile",
    settings: "/settings",
  },
};

/**
 * Build a task detail URL.
 */
export function getTaskDetailUrl(taskId: string): string {
  return `/tasks/${taskId}`;
}

/**
 * Build a project detail URL.
 */
export function getProjectDetailUrl(projectId: string): string {
  return `/projects/${projectId}`;
}
