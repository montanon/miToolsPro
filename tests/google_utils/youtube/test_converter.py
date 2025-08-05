import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from mitoolspro.google_utils.youtube.converter import video_to_audio


class TestVideoToAudio(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('mitoolspro.google_utils.youtube.converter.VideoFileClip')
    @patch('mitoolspro.google_utils.youtube.converter.logger')
    def test_video_to_audio_success(self, mock_logger, mock_video_clip):
        # Setup
        video_path = Path(self.temp_dir.name) / "test_video.mp4"
        audio_path = Path(self.temp_dir.name) / "test_audio.wav"
        
        # Create mock video clip
        mock_clip = MagicMock()
        mock_audio = MagicMock()
        mock_clip.audio = mock_audio
        mock_video_clip.return_value = mock_clip
        
        # Execute
        video_to_audio(video_path, audio_path)
        
        # Verify
        mock_video_clip.assert_called_once_with(str(video_path))
        mock_audio.write_audiofile.assert_called_once_with(str(audio_path))
        mock_clip.close.assert_called_once()
        mock_logger.info.assert_called_once_with("Audio saved to %s", audio_path)

    @patch('mitoolspro.google_utils.youtube.converter.VideoFileClip')
    @patch('mitoolspro.google_utils.youtube.converter.logger')
    def test_video_to_audio_with_string_paths(self, mock_logger, mock_video_clip):
        # Setup
        video_path = str(Path(self.temp_dir.name) / "test_video.mp4")
        audio_path = Path(self.temp_dir.name) / "test_audio.wav"
        
        # Create mock video clip
        mock_clip = MagicMock()
        mock_audio = MagicMock()
        mock_clip.audio = mock_audio
        mock_video_clip.return_value = mock_clip
        
        # Execute
        video_to_audio(video_path, audio_path)
        
        # Verify
        mock_video_clip.assert_called_once_with(video_path)
        mock_audio.write_audiofile.assert_called_once_with(str(audio_path))
        mock_clip.close.assert_called_once()
        mock_logger.info.assert_called_once_with("Audio saved to %s", audio_path)

    @patch('mitoolspro.google_utils.youtube.converter.VideoFileClip')
    @patch('mitoolspro.google_utils.youtube.converter.logger')
    def test_video_to_audio_exception_handling(self, mock_logger, mock_video_clip):
        # Setup
        video_path = Path(self.temp_dir.name) / "test_video.mp4"
        audio_path = Path(self.temp_dir.name) / "test_audio.wav"
        
        # Setup exception
        error_message = "Test error"
        mock_video_clip.side_effect = Exception(error_message)
        
        # Execute
        video_to_audio(video_path, audio_path)
        
        # Verify
        mock_logger.error.assert_called_once_with("Error converting video to audio: %s", error_message)

    @patch('mitoolspro.google_utils.youtube.converter.VideoFileClip')
    @patch('mitoolspro.google_utils.youtube.converter.logger')
    def test_video_to_audio_audio_write_exception(self, mock_logger, mock_video_clip):
        # Setup
        video_path = Path(self.temp_dir.name) / "test_video.mp4"
        audio_path = Path(self.temp_dir.name) / "test_audio.wav"
        
        # Create mock video clip that raises exception on audio write
        mock_clip = MagicMock()
        mock_audio = MagicMock()
        mock_clip.audio = mock_audio
        error_message = "Audio write failed"
        mock_audio.write_audiofile.side_effect = Exception(error_message)
        mock_video_clip.return_value = mock_clip
        
        # Execute
        video_to_audio(video_path, audio_path)
        
        # Verify
        mock_video_clip.assert_called_once_with(str(video_path))
        mock_audio.write_audiofile.assert_called_once_with(str(audio_path))
        mock_logger.error.assert_called_once_with("Error converting video to audio: %s", error_message)


if __name__ == '__main__':
    unittest.main()