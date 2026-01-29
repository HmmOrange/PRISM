export type DatasetSplit = "test" | "validation";

export interface QueryFile {
  id: string;
  file: File;
}

export interface QueryData {
  id: number;
  name: string;
  split: DatasetSplit;
  label: string;
  files: QueryFile[];
}
