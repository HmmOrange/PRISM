import { Typography, Box } from "@mui/material";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import AudioFileIcon from "@mui/icons-material/AudioFile";
import VideoFileIcon from "@mui/icons-material/VideoFile";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";

interface Props {
  file?: File | null;
  downloadUrl?: string;
  contentType?: string;
  showControls?: boolean; // For full preview with controls (default: false for thumbnails)
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function FilePreview({
  file,
  downloadUrl,
  contentType,
  showControls = false,
}: Props) {
  if (!file && !downloadUrl) {
    return (
      <Typography variant="body2" color="text.secondary">
        Select a file to preview
      </Typography>
    );
  }

  // ===== Local file (create / edit mode)
  if (file instanceof File) {
    const url = URL.createObjectURL(file);

    if (file.type.startsWith("image/")) {
      return <img src={url} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />;
    }

    if (file.type.startsWith("audio/")) {
      if (showControls) {
        return <audio controls src={url} style={{ width: "100%" }} />;
      }
      return (
        <Box sx={{ textAlign: "center", color: "text.secondary" }}>
          <AudioFileIcon sx={{ fontSize: 48 }} />
          <PlayCircleOutlineIcon sx={{ fontSize: 24, ml: -1, mt: -1 }} />
        </Box>
      );
    }

    if (file.type.startsWith("video/")) {
      if (showControls) {
        return <video controls width="100%" src={url} />;
      }
      return (
        <Box sx={{ textAlign: "center", color: "text.secondary" }}>
          <VideoFileIcon sx={{ fontSize: 48 }} />
          <PlayCircleOutlineIcon sx={{ fontSize: 24, ml: -1, mt: -1 }} />
        </Box>
      );
    }

    return (
      <Box sx={{ textAlign: "center", color: "text.secondary" }}>
        <InsertDriveFileIcon sx={{ fontSize: 48 }} />
      </Box>
    );
  }

  // ===== Remote file (view mode)
  if (downloadUrl && contentType) {
    if (contentType.startsWith("image/")) {
      return <img src={`${API_BASE}${downloadUrl}`} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />;
    }

    if (contentType.startsWith("audio/")) {
      if (showControls) {
        return <audio controls src={`${API_BASE}${downloadUrl}`} style={{ width: "100%" }} />;
      }
      return (
        <Box sx={{ textAlign: "center", color: "text.secondary" }}>
          <AudioFileIcon sx={{ fontSize: 48 }} />
          <PlayCircleOutlineIcon sx={{ fontSize: 24, ml: -1, mt: -1 }} />
        </Box>
      );
    }

    if (contentType.startsWith("video/")) {
      if (showControls) {
        return <video controls width="100%" src={`${API_BASE}${downloadUrl}`} />;
      }
      return (
        <Box sx={{ textAlign: "center", color: "text.secondary" }}>
          <VideoFileIcon sx={{ fontSize: 48 }} />
          <PlayCircleOutlineIcon sx={{ fontSize: 24, ml: -1, mt: -1 }} />
        </Box>
      );
    }

    return (
      <Box sx={{ textAlign: "center", color: "text.secondary" }}>
        <InsertDriveFileIcon sx={{ fontSize: 48 }} />
      </Box>
    );
  }

  return null;
}
 