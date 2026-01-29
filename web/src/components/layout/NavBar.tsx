import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Stack,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { ROUTES } from "../../config/routes";

export default function Navbar() {
  return (
    <AppBar position="static" color="inherit" elevation={1}>
      <Toolbar sx={{ justifyContent: "space-between" }}>
        {/* Left: App name + main nav */}
        <Stack direction="row" spacing={3} alignItems="center">
          <Typography
            variant="h6"
            component={RouterLink}
            to={ROUTES.public.home}
            sx={{
              textDecoration: "none",
              color: "inherit",
              fontWeight: 600,
            }}
          >
            PRISM
          </Typography>

          <Button
            component={RouterLink}
            to={ROUTES.public.tasks}
            color="inherit"
          >
            Tasks
          </Button>

          <Button
            component={RouterLink}
            to={ROUTES.authed.run}
            color="inherit"
            disabled
          >
            Run
          </Button>
        </Stack>

        {/* Right: Actions */}
        <Stack direction="row" spacing={1} alignItems="center">
          {/* Create Task CTA */}
          <Button
            component={RouterLink}
            to={ROUTES.public.createTask}
            variant="contained"
          >
            Create Task
          </Button>

          {/* Auth (placeholder) */}
          <Button color="inherit">
            Login
          </Button>
          <Button color="inherit">
            Register
          </Button>
        </Stack>
      </Toolbar>
    </AppBar>
  );
}
