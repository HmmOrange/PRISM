import { Stack, IconButton, Typography } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import ImageIcon from "@mui/icons-material/Image";
import AudiotrackIcon from "@mui/icons-material/Audiotrack";
import VideoFileIcon from "@mui/icons-material/VideoFile";
import DescriptionIcon from "@mui/icons-material/Description";

import type { QueryFile } from "../types";

function getIcon(file: File) {
  if (file.type.startsWith("image/"))
    return <ImageIcon fontSize="small" sx={{ color: "success.main" }} />;
  if (file.type.startsWith("audio/"))
    return <AudiotrackIcon fontSize="small" sx={{ color: "secondary.main" }} />;
  if (file.type.startsWith("video/"))
    return <VideoFileIcon fontSize="small" sx={{ color: "error.main" }} />;
  if (file.type.startsWith("text/"))
    return <DescriptionIcon fontSize="small" sx={{ color: "info.main" }} />;

  return <InsertDriveFileIcon fontSize="small" sx={{ color: "text.secondary" }} />;
}

interface Props {
  file: QueryFile;
  onDelete: () => void;
  onSelect: () => void;
  selected: boolean;
}

export default function FileItem({
  file,
  onDelete,
  onSelect,
  selected,
}: Props) {
  return (
    <Stack
      direction="row"
      spacing={1}
      alignItems="center"
      sx={{
        px: 1,
        py: 0.5,
        borderRadius: 1,
        cursor: "pointer",
        bgcolor: selected ? "action.selected" : "transparent",
      }}
      onClick={onSelect}
    >
      {getIcon(file.file)}

      <Typography variant="body2" noWrap flex={1}>
        {file.file.name}
      </Typography>

      <IconButton
        size="small"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
      >
        <DeleteIcon fontSize="small" />
      </IconButton>
    </Stack>
  );
}
