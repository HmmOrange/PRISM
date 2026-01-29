import { Outlet } from "react-router-dom";
import { Box } from "@mui/material";
import Navbar from "./NavBar";

export default function MainLayout() {
  return (
    <Box minHeight="100vh" display="flex" flexDirection="column">
      <Navbar />
      <Box component="main" flex={1}>
        <Outlet />
      </Box>
    </Box>
  );
}
