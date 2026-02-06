import { Stack, IconButton, Typography, alpha } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import ImageIcon from "@mui/icons-material/Image";
import AudiotrackIcon from "@mui/icons-material/Audiotrack";
import VideoFileIcon from "@mui/icons-material/VideoFile";
import DescriptionIcon from "@mui/icons-material/Description";
import TableChartIcon from "@mui/icons-material/TableChart";
import DataObjectIcon from "@mui/icons-material/DataObject";

import type {
  QueryFile,
  LocalQueryFile,
} from "../../../types/tasks.types";

type AnyQueryFile = QueryFile | LocalQueryFile;

function isLocalFile(file: AnyQueryFile): file is LocalQueryFile {
  return "file" in file;
}

function getIconByMime(mime?: string, filename?: string) {
  // Check by extension first
  const ext = filename?.split(".").pop()?.toLowerCase();
  
  if (ext === "csv" || mime === "text/csv") {
    return <TableChartIcon fontSize="small" sx={{ color: "success.main" }} />;
  }
  
  if (ext === "json" || mime === "application/json") {
    return <DataObjectIcon fontSize="small" sx={{ color: "warning.main" }} />;
  }
  
  if (!mime) {
    return (
      <InsertDriveFileIcon
        fontSize="small"
        sx={{ color: "text.secondary" }}
      />
    );
  }

  if (mime.startsWith("image/"))
    return <ImageIcon fontSize="small" sx={{ color: "info.main" }} />;

  if (mime.startsWith("audio/"))
    return <AudiotrackIcon fontSize="small" sx={{ color: "secondary.main" }} />;

  if (mime.startsWith("video/"))
    return <VideoFileIcon fontSize="small" sx={{ color: "error.main" }} />;

  if (mime.startsWith("text/"))
    return <DescriptionIcon fontSize="small" sx={{ color: "primary.main" }} />;

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
        px: 1.5,
        py: 0.75,
        borderRadius: 1,
        cursor: "pointer",
        bgcolor: selected ? (theme) => alpha(theme.palette.primary.main, 0.12) : "transparent",
        border: "1px solid",
        borderColor: selected ? "primary.main" : "transparent",
        "&:hover": {
          bgcolor: selected
            ? (theme) => alpha(theme.palette.primary.main, 0.12)
            : "action.hover",
        },
        "&:hover .delete-btn": {
          opacity: 1,
        },
        transition: "all 0.15s ease",
      }}
      onClick={onSelect}
    >
      {getIconByMime(mime, filename)}

      <Typography
        variant="body2"
        noWrap
        flex={1}
        sx={{
          fontWeight: selected ? 500 : 400,
        }}
      >
        {filename}
      </Typography>

      {!readOnly && onDelete && (
        <IconButton
          className="delete-btn"
          size="small"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          sx={{
            opacity: 0,
            transition: "opacity 0.15s ease",
            "&:hover": {
              color: "error.main",
            },
          }}
        >
          <DeleteIcon fontSize="small" />
        </IconButton>
      )}
    </Stack>
  );
}
