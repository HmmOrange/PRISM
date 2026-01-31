import {
  Snackbar,
  Alert,
  type AlertColor,
} from "@mui/material";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";


interface Toast {
  message: string;
  severity?: AlertColor;
}

interface ToastContextValue {
  showToast: (toast: Toast) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [toast, setToast] = useState<Toast>({
    message: "",
    severity: "success",
  });

  function showToast({ message, severity = "success" }: Toast) {
    setToast({ message, severity });
    setOpen(true);
  }

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}

      <Snackbar
        open={open}
        autoHideDuration={3000}
        onClose={() => setOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        <Alert
          onClose={() => setOpen(false)}
          severity={toast.severity}
          variant="filled"
        >
          {toast.message}
        </Alert>
      </Snackbar>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within ToastProvider");
  }
  return ctx;
}
