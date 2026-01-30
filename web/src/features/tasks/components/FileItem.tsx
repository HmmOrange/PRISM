import { Stack, IconButton, Typography } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import ImageIcon from "@mui/icons-material/Image";
import AudiotrackIcon from "@mui/icons-material/Audiotrack";
import VideoFileIcon from "@mui/icons-material/VideoFile";
import DescriptionIcon from "@mui/icons-material/Description";

import type {
  QueryFile,
  LocalQueryFile,
} from "../../../types/tasks.types";

type AnyQueryFile = QueryFile | LocalQueryFile;

function isLocalFile(file: AnyQueryFile): file is LocalQueryFile {
  return "file" in file;
}

function getIconByMime(mime?: string) {
  if (!mime) {
    return (
      <InsertDriveFileIcon
        fontSize="small"
        sx={{ color: "text.secondary" }}
      />
    );
  }

  if (mime.startsWith("image/"))
    return <ImageIcon fontSize="small" sx={{ color: "success.main" }} />;

  if (mime.startsWith("audio/"))
    return <AudiotrackIcon fontSize="small" sx={{ color: "secondary.main" }} />;

  if (mime.startsWith("video/"))
    return <VideoFileIcon fontSize="small" sx={{ color: "error.main" }} />;

  if (mime.startsWith("text/"))
    return <DescriptionIcon fontSize="small" sx={{ color: "info.main" }} />;

  return (
    <InsertDriveFileIcon
      fontSize="small"
      sx={{ color: "text.secondary" }}
    />
  );
}

interface Props {
  file: AnyQueryFile;
  onDelete?: () => void;
  onSelect: () => void;
  selected: boolean;
  readOnly?: boolean;
}

export default function FileItem({
  file,
  onDelete,
  onSelect,
  selected,
  readOnly = false,
}: Props) {
  const mime = isLocalFile(file)
    ? file.file.type
    : file.content_type;

  const filename = isLocalFile(file)
    ? file.file.name
    : file.filename;

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
        "&:hover": {
          bgcolor: selected
            ? "action.selected"
            : "action.hover",
        },
      }}
      onClick={onSelect}
    >
      {getIconByMime(mime)}

      <Typography variant="body2" noWrap flex={1}>
        {filename}
      </Typography>

      {!readOnly && onDelete && (
        <IconButton
          size="small"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
        >
          <DeleteIcon fontSize="small" />
        </IconButton>
      )}
    </Stack>
  );
}
