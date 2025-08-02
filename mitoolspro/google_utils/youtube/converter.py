from os import PathLike
from pathlib import Path

from moviepy import VideoFileClip
import logging

logger = logging.getLogger(__name__)


def video_to_audio(video_path: PathLike, audio_path: Path):
    try:
        video_path, audio_path = Path(video_path), Path(audio_path)
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path))
        clip.close()
        logger.info("Audio saved to %s", audio_path)
    except Exception as e:
        logger.error("Error converting video to audio: %s", str(e))
