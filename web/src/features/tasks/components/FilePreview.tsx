import { Typography } from "@mui/material";

interface Props {
  file?: File | null;
  downloadUrl?: string;
  contentType?: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function FilePreview({
  file,
  downloadUrl,
  contentType,
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
      return <img src={url} style={{ maxWidth: "100%" }} />;
    }

    if (file.type.startsWith("audio/")) {
      return <audio controls src={url} />;
    }

    if (file.type.startsWith("video/")) {
      return <video controls width="100%" src={url} />;
    }

    return (
      <Typography variant="body2">
        Preview not available for this file type
      </Typography>
    );
  }

  // ===== Remote file (view mode)
  if (downloadUrl && contentType) {
    if (contentType.startsWith("image/")) {
      return <img src={`${API_BASE}${downloadUrl}`} style={{ maxWidth: "100%" }} />;
    }

    if (contentType.startsWith("audio/")) {
      return <audio controls src={`${API_BASE}${downloadUrl}`} />;
    }

    if (contentType.startsWith("video/")) {
      return <video controls width="100%" src={`${API_BASE}${downloadUrl}`} />;
    }

    return (
      <Typography variant="body2">
        Preview not available for this file type
      </Typography>
    );
  }

  return null;
}
 