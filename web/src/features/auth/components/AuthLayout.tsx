/**
 * Auth Layout component.
 * Implements SRS 2.1.2 Split Layout Design:
 * - Visual Panel: Brand logo (logotype-colored.svg) and marketing copy
 * - Interaction Panel: Centered form area
 *
 * Uses the PRISM design-system palette (#25343F, #FF9B51, #EAEFEF, #BFC9D1).
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
      {/* Visual Panel — Brand Section */}
      <Box
        sx={{
          flex: isMobile ? "none" : 1,
          minHeight: isMobile ? 220 : "100vh",
          bgcolor: "#25343F",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          p: { xs: 4, md: 6 },
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Subtle dot pattern background */}
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            opacity: 0.04,
            backgroundImage:
              "radial-gradient(circle, #EAEFEF 1px, transparent 1px)",
            backgroundSize: "28px 28px",
          }}
        />

        {/* Decorative accent circle */}
        <Box
          sx={{
            position: "absolute",
            top: -80,
            right: -80,
            width: 260,
            height: 260,
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(255,155,81,0.18) 0%, transparent 70%)",
          }}
        />
        <Box
          sx={{
            position: "absolute",
            bottom: -100,
            left: -60,
            width: 300,
            height: 300,
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(255,155,81,0.10) 0%, transparent 70%)",
          }}
        />

        {/* Content */}
        <Box
          sx={{
            position: "relative",
            zIndex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            textAlign: "center",
            maxWidth: 420,
          }}
        >
          {/* Logo SVG */}
          <Box
            component="img"
            src="/logotype-white-colored.svg"
            alt="PRISM"
            sx={{
              width: { xs: 180, md: 260 },
              mb: 4,
              filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.3))",
            }}
          />

          <Typography
            variant="h5"
            sx={{
              color: "#EAEFEF",
              fontWeight: 500,
              mb: 2,
              letterSpacing: "0.02em",
            }}
          >
            Model-Based Workflow Generation
          </Typography>

          <Typography
            variant="body1"
            sx={{
              color: "#BFC9D1",
              lineHeight: 1.7,
              maxWidth: 360,
            }}
          >
            Create complex task workflows utilizing optimal models for your
            machine learning and AI projects.
          </Typography>
        </Box>
      </Box>

      {/* Interaction Panel — Form Section */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#EAEFEF",
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
