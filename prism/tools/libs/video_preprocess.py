import os
from pathlib import Path
from typing import Optional, List
import warnings
import numpy as np 
from PIL import Image

from prism.tools.tool_registry import register_tool

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Image processing tools will be limited.")

try:
    import moviepy.editor as mp
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    warnings.warn("moviepy not available. Video processing tools will not work.")


VIDEO_TAGS = ["video processing", "multimedia"]

@register_tool(tags=VIDEO_TAGS)
class VideoFrameExtractor:
    """
    Extract frames from video files.
    """
    
    def extract_frames(self, video_path: str, frame_time: Optional[float] = None) -> List[Image.Image]:
        """
        Extract a single frame from video (instead of all frames).

        Args:
            video_path (str): Path to input video
            frame_time (Optional[float]): Time (in seconds) to extract frame.
                                          If None, defaults to the middle of the video.

        Returns:
            List[Image.Image]: A list containing one extracted frame as PIL Image.
        """
        if not HAS_MOVIEPY and not HAS_CV2:
            raise ImportError("Either moviepy or OpenCV must be installed for video processing")
            
        frames = []
        
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Xác định frame index
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if frame_time is None:  # lấy frame giữa video
                frame_time = duration / 2.0
            frame_idx = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            cap.release()
        
        elif HAS_MOVIEPY:
            clip = mp.VideoFileClip(video_path)
            duration = clip.duration
            if frame_time is None:  # mặc định frame giữa
                frame_time = duration / 2.0
            frame = clip.get_frame(frame_time)
            frames.append(Image.fromarray(frame.astype("uint8")))
            clip.close()
        
        return frames


@register_tool(tags=VIDEO_TAGS)
class VideoAudioExtractor:
    """
    Extract audio from video files.
    """
    
    def extract_audio(self, video_path: str, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Extract audio from video.
        
        Args:
            video_path (str): Path to input video
            sample_rate (Optional[int]): Target sample rate (if None, use original)
            
        Returns:
            np.ndarray: Extracted audio as numpy array
        """
        if not HAS_MOVIEPY:
            raise ImportError("moviepy is required for video audio extraction")
            
        clip = mp.VideoFileClip(video_path)
        
        if clip.audio is None:
            raise ValueError(f"No audio track found in video: {video_path}")
        
        # Get audio array
        audio_array = clip.audio.to_soundarray()
        
        # If stereo, convert to mono by averaging channels
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample if needed
        if sample_rate is not None and hasattr(clip.audio, 'fps') and clip.audio.fps != sample_rate:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=int(clip.audio.fps), target_sr=sample_rate)
            except ImportError:
                print(f"Warning: librosa not available. Cannot resample to {sample_rate}Hz")
        
        clip.close()
        
        return audio_array
