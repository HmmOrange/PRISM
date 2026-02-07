/**
 * Task Wizard Hook.
 * Manages state for the multi-step task creation wizard.
 * Implements SRS 2.3.1 Create Task Wizard
 */

import { useState, useCallback } from "react";
import type { EditableQuery } from "../../../types/tasks.types";

export interface TaskWizardState {
  /** Current step (0-3) */
  activeStep: number;

  /** Stage 1: Metadata */
  name: string;
  description: string;
  metrics: string[];

  /** Stage 2: Pipeline Tags */
  pipelineTags: string[];

  /** Stage 3: Dataset/Queries */
  queries: EditableQuery[];

  /** Submission state */
  isSubmitting: boolean;
  submitError: string | null;
}

interface ValidationErrors {
  name?: string;
  metrics?: string;
  pipelineTags?: string;
  queries?: string;
}

const TOTAL_STEPS = 4;

const INITIAL_STATE: TaskWizardState = {
  activeStep: 0,
  name: "",
  description: "",
  metrics: [],
  pipelineTags: [],
  queries: [],
  isSubmitting: false,
  submitError: null,
};

export function useTaskWizard() {
  const [state, setState] = useState<TaskWizardState>(INITIAL_STATE);
  const [errors, setErrors] = useState<ValidationErrors>({});

  /**
   * Validate current step.
   */
  const validateStep = useCallback((step: number): boolean => {
    const newErrors: ValidationErrors = {};

    switch (step) {
      case 0: // Metadata
        if (!state.name.trim()) {
          newErrors.name = "Task name is required";
        }
        if (state.metrics.length === 0) {
          newErrors.metrics = "Please select at least one metric";
        }
        break;

      case 1: // Pipeline Tags
        // Pipeline tags are optional, no validation needed
        break;

      case 2: // Dataset
        // Queries are optional but if added, should have files or labels
        break;

      case 3: // Review
        // All validation should have been done in previous steps
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [state]);

  /**
   * Go to next step.
   */
  const nextStep = useCallback(() => {
    if (!validateStep(state.activeStep)) {
      return false;
    }

    if (state.activeStep < TOTAL_STEPS - 1) {
      setState((prev) => ({ ...prev, activeStep: prev.activeStep + 1 }));
      return true;
    }
    return false;
  }, [state.activeStep, validateStep]);

  /**
   * Go to previous step.
   */
  const prevStep = useCallback(() => {
    if (state.activeStep > 0) {
      setState((prev) => ({ ...prev, activeStep: prev.activeStep - 1 }));
      setErrors({});
    }
  }, [state.activeStep]);

  /**
   * Go to specific step.
   */
  const goToStep = useCallback((step: number) => {
    if (step >= 0 && step < TOTAL_STEPS && step <= state.activeStep) {
      setState((prev) => ({ ...prev, activeStep: step }));
      setErrors({});
    }
  }, [state.activeStep]);

  /**
   * Update metadata fields.
   */
  const setMetadata = useCallback((updates: Partial<Pick<TaskWizardState, "name" | "description" | "metrics">>) => {
    setState((prev) => ({ ...prev, ...updates }));
    // Clear related errors
    if (updates.name !== undefined && errors.name) {
      setErrors((prev) => ({ ...prev, name: undefined }));
    }
    if (updates.metrics !== undefined && errors.metrics) {
      setErrors((prev) => ({ ...prev, metrics: undefined }));
    }
  }, [errors]);

  /**
   * Update pipeline tags.
   */
  const setPipelineTags = useCallback((tags: string[]) => {
    setState((prev) => ({ ...prev, pipelineTags: tags }));
    if (errors.pipelineTags) {
      setErrors((prev) => ({ ...prev, pipelineTags: undefined }));
    }
  }, [errors]);

  /**
   * Toggle a pipeline tag.
   */
  const togglePipelineTag = useCallback((tag: string) => {
    setState((prev) => ({
      ...prev,
      pipelineTags: prev.pipelineTags.includes(tag)
        ? prev.pipelineTags.filter((t) => t !== tag)
        : [...prev.pipelineTags, tag],
    }));
    if (errors.pipelineTags) {
      setErrors((prev) => ({ ...prev, pipelineTags: undefined }));
    }
  }, [errors]);

  /**
   * Update queries.
   */
  const setQueries = useCallback((queries: EditableQuery[]) => {
    setState((prev) => ({ ...prev, queries }));
  }, []);

  /**
   * Add a new query.
   */
  const addQuery = useCallback(() => {
    setState((prev) => ({
      ...prev,
      queries: [
        ...prev.queries,
        {
          id: prev.queries.length,
          name: "",
          split: "test",
          label: "",
          files: [],
        },
      ],
    }));
  }, []);

  /**
   * Update a specific query.
   */
  const updateQuery = useCallback((updated: EditableQuery) => {
    setState((prev) => ({
      ...prev,
      queries: prev.queries.map((q) => (q.id === updated.id ? updated : q)),
    }));
  }, []);

  /**
   * Remove a query.
   */
  const removeQuery = useCallback((queryId: number) => {
    setState((prev) => ({
      ...prev,
      queries: prev.queries
        .filter((q) => q.id !== queryId)
        .map((q, idx) => ({ ...q, id: idx })), // Reindex
    }));
  }, []);

  /**
   * Set submission state.
   */
  const setSubmitting = useCallback((isSubmitting: boolean, error?: string | null) => {
    setState((prev) => ({
      ...prev,
      isSubmitting,
      submitError: error ?? null,
    }));
  }, []);

  /**
   * Reset wizard to initial state.
   */
  const reset = useCallback(() => {
    setState(INITIAL_STATE);
    setErrors({});
  }, []);

  /**
   * Check if can proceed to next step.
   */
  const canProceed = useCallback((): boolean => {
    switch (state.activeStep) {
      case 0:
        return state.name.trim() !== "" && state.metrics.length > 0;
      case 1:
        return true; // Pipeline tags are optional
      case 2:
        return true; // Queries are optional
      case 3:
        return true; // Review step
      default:
        return false;
    }
  }, [state]);

  return {
    state,
    errors,
    totalSteps: TOTAL_STEPS,

    // Navigation
    nextStep,
    prevStep,
    goToStep,
    canProceed,

    // Metadata
    setMetadata,

    // Pipeline Tags
    setPipelineTags,
    togglePipelineTag,

    // Queries
    setQueries,
    addQuery,
    updateQuery,
    removeQuery,

    // Submission
    setSubmitting,

    // Utils
    reset,
    validateStep,
  };
}
