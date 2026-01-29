import { RouterProvider } from "react-router-dom";
import { router } from "../routes/router";
import { AppProviders } from "./AppProviders";

export default function App() {
  return (
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>
  );
}
