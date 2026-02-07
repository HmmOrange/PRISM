/**
 * Navigation Bar component.
 * Implements SRS 2.2 Navigation requirements:
 * - Home
 * - Task Library
 * - Project Library (placeholder)
 * - User Profile/Settings/Logout
 */

import { useState } from "react";
import {
  AppBar,
  Toolbar,
  Button,
  Stack,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Box,
  useTheme,
  useMediaQuery,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import { Link as RouterLink, useLocation, useNavigate } from "react-router-dom";
import MenuIcon from "@mui/icons-material/Menu";
import DashboardIcon from "@mui/icons-material/Dashboard";
import AssignmentIcon from "@mui/icons-material/Assignment";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PersonIcon from "@mui/icons-material/Person";
import LogoutIcon from "@mui/icons-material/Logout";

import { ROUTES } from "../../config/routes";

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", path: ROUTES.public.dashboard, icon: <DashboardIcon /> },
  { label: "Tasks", path: ROUTES.public.tasks, icon: <AssignmentIcon /> },
  { label: "Run", path: ROUTES.public.runTasks, icon: <PlayArrowIcon /> },
  // { label: "Projects", path: ROUTES.public.projects, icon: <FolderIcon /> },
];

export default function Navbar() {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  function isActive(path: string): boolean {
    if (path === ROUTES.public.dashboard) {
      return location.pathname === path || location.pathname === "/";
    }
    return location.pathname.startsWith(path);
  }

  function handleProfileClick(event: React.MouseEvent<HTMLElement>) {
    setAnchorEl(event.currentTarget);
  }

  function handleProfileClose() {
    setAnchorEl(null);
  }

  function handleLogout() {
    handleProfileClose();
    // Logout will be handled by auth context when implemented
    navigate(ROUTES.public.login);
  }

  // Mobile drawer content
  const drawer = (
    <Box sx={{ width: 250 }} onClick={() => setMobileOpen(false)}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: "divider" }}>
        <img src="/logotype-colored.svg" alt="PRISM" style={{ height: 24 }} />
      </Box>
      <List>
        {NAV_ITEMS.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={RouterLink}
              to={item.path}
              selected={isActive(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton component={RouterLink} to={ROUTES.public.createTask}>
            <ListItemText primary="Create Task" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <>
      <AppBar position="sticky" color="inherit" elevation={1}>
        <Toolbar sx={{ justifyContent: "space-between" }}>
          {/* Mobile menu button */}
          {isMobile && (
            <IconButton
              edge="start"
              onClick={() => setMobileOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          {/* Logo */}
          <Box
            component={RouterLink}
            to={ROUTES.public.home}
            sx={{
              display: "flex",
              alignItems: "center",
              textDecoration: "none",
              mr: 4,
              "& img": {
                height: 50,
                transition: "opacity 0.2s",
              },
              "& .logo-default": {
                display: "block",
              },
              "& .logo-hover": {
                display: "none",
              },
              "&:hover .logo-default": {
                display: "none",
              },
              "&:hover .logo-hover": {
                display: "block",
              },
            }}
          >
            <img src="/logotype.svg" alt="PRISM" className="logo-default" />
            <img src="/logotype-colored.svg" alt="PRISM" className="logo-hover" />
          </Box>

          {/* Desktop Navigation */}
          {!isMobile && (
            <Stack direction="row" spacing={1} flex={1}>
              {NAV_ITEMS.map((item) => (
                <Button
                  key={item.path}
                  component={RouterLink}
                  to={item.path}
                  color="inherit"
                  sx={{
                    fontWeight: isActive(item.path) ? 600 : 400,
                    borderRadius: 0,
                    py: 2,
                    my: -2,
                    px: 2,
                    position: "relative",
                    "&::after": isActive(item.path) ? {
                      content: '""',
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      bgcolor: "secondary.main",
                    } : {},
                    color: isActive(item.path) ? "secondary.main" : "inherit",
                    "&:hover": {
                      color: "secondary.main",
                    },
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </Stack>
          )}

          {/* Right Actions */}
          <Stack direction="row" spacing={1} alignItems="center">
            {!isMobile && (
              <Button
                component={RouterLink}
                to={ROUTES.public.createTask}
                variant="contained"
              >
                Create Task
              </Button>
            )}

            {/* Profile Menu */}
            <IconButton onClick={handleProfileClick} size="small">
              <PersonIcon />
            </IconButton>
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleProfileClose}
              anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
              transformOrigin={{ vertical: "top", horizontal: "right" }}
              disableScrollLock
            >
              <MenuItem
                component={RouterLink}
                to={ROUTES.public.login}
                onClick={handleProfileClose}
              >
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                Login
              </MenuItem>
              <MenuItem
                component={RouterLink}
                to={ROUTES.public.register}
                onClick={handleProfileClose}
              >
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                Register
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </Stack>
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{ keepMounted: true }}
      >
        {drawer}
      </Drawer>
    </>
  );
}
