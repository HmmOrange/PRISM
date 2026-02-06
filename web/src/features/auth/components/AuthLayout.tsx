/**
 * Auth Layout component.
 * Implements SRS 2.1.2 Split Layout Design:
 * - Visual Panel: Brand logo and marketing copy
 * - Interaction Panel: Centered form area
 */

import { Box, Container, Typography, useTheme, useMediaQuery } from "@mui/material";
import type { ReactNode } from "react";

interface AuthLayoutProps {
  children: ReactNode;
}

export default function AuthLayout({ children }: AuthLayoutProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Box
      display="flex"
      minHeight="100vh"
      flexDirection={isMobile ? "column" : "row"}
    >
      {/* Visual Panel - Brand Section */}
      <Box
        sx={{
          flex: isMobile ? "none" : 1,
          minHeight: isMobile ? 200 : "100vh",
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          color: "white",
          p: 4,
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Background pattern */}
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            opacity: 0.1,
            backgroundImage: `radial-gradient(circle at 25% 25%, white 2%, transparent 2%),
                            radial-gradient(circle at 75% 75%, white 2%, transparent 2%)`,
            backgroundSize: "60px 60px",
          }}
        />

        <Box sx={{ position: "relative", zIndex: 1, textAlign: "center" }}>
          <Typography
            variant="h2"
            fontWeight={700}
            sx={{ mb: 2 }}
          >
            PRISM
          </Typography>
          <Typography
            variant="h5"
            sx={{ mb: 3, opacity: 0.9 }}
          >
            Model-Based Workflow Generation
          </Typography>
          <Typography
            variant="body1"
            sx={{ maxWidth: 400, opacity: 0.8, lineHeight: 1.6 }}
          >
            Create complex task workflows utilizing optimal models
            for your machine learning and AI projects.
          </Typography>
        </Box>

        {/* Decorative elements */}
        <Box
          sx={{
            position: "absolute",
            bottom: -100,
            right: -100,
            width: 300,
            height: 300,
            borderRadius: "50%",
            background: "rgba(255,255,255,0.1)",
          }}
        />
        <Box
          sx={{
            position: "absolute",
            top: -50,
            left: -50,
            width: 200,
            height: 200,
            borderRadius: "50%",
            background: "rgba(255,255,255,0.05)",
          }}
        />
      </Box>

      {/* Interaction Panel - Form Section */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "background.default",
          p: { xs: 3, md: 6 },
        }}
      >
        <Container maxWidth="sm" sx={{ maxWidth: 440 }}>
          {children}
        </Container>
      </Box>
    </Box>
  );
}
