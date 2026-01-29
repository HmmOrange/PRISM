import { createBrowserRouter } from "react-router-dom";
import { publicRoutes } from "./public.routes";
import { authedRoutes } from "./authed.routes";

export const router = createBrowserRouter([
  ...publicRoutes,
  ...authedRoutes,
]);
