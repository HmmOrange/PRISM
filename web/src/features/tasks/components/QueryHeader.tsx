import { Box, Typography, Chip, IconButton } from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import type { QueryData } from "../types";

interface Props {
  query: QueryData;
  onRemove: () => void;
}

export default function QueryHeader({ query, onRemove }: Props) {
  const missingLabel = !query.label?.trim();

  return (
    <Box
      display="flex"
      alignItems="center"
      width="99%"
      justifyContent="space-between"
    >
      {/* Left: name */}
      <Typography fontWeight={600}>
        {query.name || `Query ${query.id}`}
      </Typography>

      {/* Right: badges + delete */}
      <Box display="flex" alignItems="center" gap={1}>
        {missingLabel && (
          <Chip
            label="Missing label"
            color="warning"
            size="small"
            variant="outlined"
          />
        )}

        <Chip
          label={`${query.files.length} file${
            query.files.length !== 1 ? "s" : ""
          }`}
          size="small"
          variant="outlined"
        />

        <IconButton
          size="small"
          color="error"
          onClick={(e) => {
            e.stopPropagation(); // 🔑 prevent accordion toggle
            onRemove();
          }}
        >
          <DeleteOutlineIcon fontSize="small" />
        </IconButton>
      </Box>
    </Box>
  );
}
