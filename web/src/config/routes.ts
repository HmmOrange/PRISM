export const ROUTES = {
  public: {
    home: "/",
    tasks: "/tasks",
    createTask: "/tasks/new",
    login: "/login",
    register: "/register",
    taskDetail: "/tasks/:taskId",
    runTasks: "/run",
  },
  authed: {
    tasks: "/tasks",
    run: "/run",
  },
};
