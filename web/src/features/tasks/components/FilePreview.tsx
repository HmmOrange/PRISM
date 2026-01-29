import { Box, Typography } from "@mui/material";

interface Props {
  file?: File;
}

export default function FilePreview({ file }: Props) {
  if (!file) {
    return (
      <Typography variant="body2" color="text.secondary">
        Select a file to preview
      </Typography>
    );
  }

  if (file.type.startsWith("image/")) {
    return <img src={URL.createObjectURL(file)} style={{ maxWidth: "100%" }} />;
  }

  if (file.type.startsWith("audio/")) {
    return <audio controls src={URL.createObjectURL(file)} />;
  }

  if (file.type.startsWith("video/")) {
    return <video controls width="100%" src={URL.createObjectURL(file)} />;
  }

  return (
    <Box>
      <Typography variant="body2">
        Preview not available for this file type
      </Typography>
    </Box>
  );
}
