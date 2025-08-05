import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.google_utils.youtube.downloader import (
    download_video,
    download_audio_video,
    batch_download,
)


class TestYouTubeDownloader(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    @patch('mitoolspro.google_utils.youtube.downloader.logger')
    def test_download_video_success(self, mock_logger, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name) / "video.mp4"
        
        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_stream = Mock()
        mock_yt.streams.filter.return_value.first.return_value = mock_stream
        mock_youtube.return_value = mock_yt
        
        # Execute
        download_video(self.test_url, output_path, resolution="720p")
        
        # Verify
        mock_youtube.assert_called_once_with(self.test_url)
        mock_yt.streams.filter.assert_called_once_with(res="720p", file_extension="mp4")
        mock_stream.download.assert_called_once_with(output_path)
        mock_logger.info.assert_any_call("Downloading: %s", "Test Video")
        mock_logger.info.assert_any_call("Download completed successfully!")

    def test_download_video_file_exists_no_recalculate(self):
        # Setup
        output_path = Path(self.temp_dir.name) / "existing_video.mp4"
        output_path.touch()  # Create the file
        
        # Execute and verify exception
        with self.assertRaises(ArgumentValueError) as context:
            download_video(self.test_url, output_path, recalculate=False)
        
        self.assertEqual(str(context.exception), "Output path already exists.")

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    def test_download_video_file_exists_with_recalculate(self, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name) / "existing_video.mp4"
        output_path.touch()  # Create the file
        
        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_stream = Mock()
        mock_yt.streams.filter.return_value.first.return_value = mock_stream
        mock_youtube.return_value = mock_yt
        
        # Execute - should not raise exception
        download_video(self.test_url, output_path, recalculate=True)
        
        # Verify download was called
        mock_stream.download.assert_called_once_with(output_path)

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    def test_download_video_resolution_not_available(self, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name) / "video.mp4"
        
        mock_yt = Mock()
        mock_yt.streams.filter.return_value.first.return_value = None  # No stream available
        mock_youtube.return_value = mock_yt
        
        # Execute and verify exception
        with self.assertRaises(RuntimeError) as context:
            download_video(self.test_url, output_path, resolution="4K")
        
        self.assertEqual(str(context.exception), "Error downloading video: Resolution 4K not available.")

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    def test_download_video_general_exception(self, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name) / "video.mp4"
        
        mock_youtube.side_effect = Exception("YouTube API error")
        
        # Execute and verify exception
        with self.assertRaises(RuntimeError) as context:
            download_video(self.test_url, output_path)
        
        self.assertEqual(str(context.exception), "Error downloading video: YouTube API error")

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    @patch('mitoolspro.google_utils.youtube.downloader.logger')
    def test_download_audio_video_success(self, mock_logger, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name)
        
        mock_yt = Mock()
        mock_yt.title = "Test Audio"
        mock_stream = Mock()
        mock_yt.streams.filter.return_value.first.return_value = mock_stream
        mock_youtube.return_value = mock_yt
        
        # Execute
        download_audio_video(self.test_url, output_path)
        
        # Verify
        mock_youtube.assert_called_once_with(self.test_url)
        mock_yt.streams.filter.assert_called_once_with(only_audio=True)
        mock_stream.download.assert_called_once_with(output_path)
        mock_logger.info.assert_any_call("Downloading: %s", "Test Audio")
        mock_logger.info.assert_any_call("Download completed successfully!")

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    def test_download_audio_video_no_stream_available(self, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name)
        
        mock_yt = Mock()
        mock_yt.streams.filter.return_value.first.return_value = None
        mock_youtube.return_value = mock_yt
        
        # Execute and verify exception
        with self.assertRaises(RuntimeError) as context:
            download_audio_video(self.test_url, output_path)
        
        self.assertEqual(str(context.exception), "Error downloading audio: Couldn't get audio stream.")

    @patch('mitoolspro.google_utils.youtube.downloader.YouTube')
    def test_download_audio_video_general_exception(self, mock_youtube):
        # Setup
        output_path = Path(self.temp_dir.name)
        
        mock_youtube.side_effect = Exception("Connection error")
        
        # Execute and verify exception
        with self.assertRaises(RuntimeError) as context:
            download_audio_video(self.test_url, output_path)
        
        self.assertEqual(str(context.exception), "Error downloading audio: Connection error")

    @patch('mitoolspro.google_utils.youtube.downloader.download_video')
    @patch('mitoolspro.google_utils.youtube.downloader.logger')
    def test_batch_download_success(self, mock_logger, mock_download_video):
        # Setup
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=abc123",
        ]
        output_path = Path(self.temp_dir.name)
        
        # Execute
        batch_download(urls, output_path, resolution="720p")
        
        # Verify that download_video was called for each URL
        self.assertEqual(mock_download_video.call_count, len(urls))

    @patch('mitoolspro.google_utils.youtube.downloader.download_video')
    @patch('mitoolspro.google_utils.youtube.downloader.logger')
    def test_batch_download_with_errors(self, mock_logger, mock_download_video):
        # Setup
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=invalid",
        ]
        output_path = Path(self.temp_dir.name)
        
        # Setup one download to fail
        mock_download_video.side_effect = [None, Exception("Download failed")]
        
        # Execute - should not raise exception, but log errors
        batch_download(urls, output_path)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()

    @patch('mitoolspro.google_utils.youtube.downloader.download_video')
    def test_batch_download_with_custom_resolution(self, mock_download_video):
        # Setup
        urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        output_path = Path(self.temp_dir.name)
        
        # Execute
        batch_download(urls, output_path, resolution="1080p")
        
        # Verify download_video was called with correct resolution
        mock_download_video.assert_called_once_with(urls[0], output_path, "1080p")


if __name__ == '__main__':
    unittest.main()