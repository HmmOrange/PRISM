import os
from pathlib import Path
from typing import List, Optional, Union
import warnings

import numpy as np

from prism.tools.tool_registry import register_tool

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    warnings.warn("librosa/soundfile not available. Audio processing tools will not work.")

AUDIO_TAGS = ["audio processing", "signal processing"]

@register_tool(tags=AUDIO_TAGS)
class AudioSplitter:
    """
    Split audio files into segments.
    """
    
    def split_audio(self, audio_path: str, segment_duration: float = 10.0, overlap: float = 0.0) -> List[np.ndarray]:
        """
        Split audio into segments.
        
        Args:
            audio_path (str): Path to input audio file
            segment_duration (float): Duration of each segment in seconds
            overlap (float): Overlap between segments in seconds
            
        Returns:
            List[np.ndarray]: List of audio segments as numpy arrays
        """
        if not HAS_AUDIO:
            raise ImportError("librosa and soundfile are required for audio processing")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        segment_samples = int(segment_duration * sr)
        overlap_samples = int(overlap * sr)
        step_samples = segment_samples - overlap_samples
        
        segments = []
        
        for i, start in enumerate(range(0, len(y) - segment_samples + 1, step_samples)):
            end = start + segment_samples
            segment = y[start:end]
            segments.append(segment)
        
        return segments

    