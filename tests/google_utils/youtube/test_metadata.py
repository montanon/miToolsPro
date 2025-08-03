import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from mitoolspro.google_utils.youtube.metadata import extract_metadata, save_metadata


class TestYouTubeMetadata(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('mitoolspro.google_utils.youtube.metadata.YouTube')
    def test_extract_metadata_success(self, mock_youtube):
        # Setup mock YouTube object
        mock_yt = Mock()
        mock_yt.title = "Test Video Title"
        mock_yt.description = "Test description"
        mock_yt.length = 210  # 3 minutes 30 seconds
        mock_yt.views = 1000000
        mock_yt.author = "Test Author"
        mock_yt.channel_id = "UCtest123"
        mock_yt.channel_url = "https://www.youtube.com/channel/UCtest123"
        mock_yt.keywords = ["test", "video", "sample"]
        mock_yt.metadata = {"category": "Entertainment"}
        mock_yt.publish_date = datetime(2023, 1, 15)
        mock_yt.rating = 4.5
        
        mock_youtube.return_value = mock_yt
        
        # Execute
        result = extract_metadata(self.test_url)
        
        # Verify
        mock_youtube.assert_called_once_with(self.test_url)
        
        expected_metadata = {
            "title": "Test Video Title",
            "description": "Test description",
            "length": 210,
            "views": 1000000,
            "author": "Test Author",
            "channel_id": "UCtest123",
            "channel_url": "https://www.youtube.com/channel/UCtest123",
            "keywords": ["test", "video", "sample"],
            "metadata": {"category": "Entertainment"},
            "publish_date": datetime(2023, 1, 15),
            "rating": 4.5,
        }
        
        self.assertEqual(result, expected_metadata)

    @patch('mitoolspro.google_utils.youtube.metadata.YouTube')
    def test_extract_metadata_with_none_values(self, mock_youtube):
        # Setup mock YouTube object with None values
        mock_yt = Mock()
        mock_yt.title = None
        mock_yt.description = None
        mock_yt.length = None
        mock_yt.views = None
        mock_yt.author = None
        mock_yt.channel_id = None
        mock_yt.channel_url = None
        mock_yt.keywords = None
        mock_yt.metadata = None
        mock_yt.publish_date = None
        mock_yt.rating = None
        
        mock_youtube.return_value = mock_yt
        
        # Execute
        result = extract_metadata(self.test_url)
        
        # Verify
        expected_metadata = {
            "title": None,
            "description": None,
            "length": None,
            "views": None,
            "author": None,
            "channel_id": None,
            "channel_url": None,
            "keywords": None,
            "metadata": None,
            "publish_date": None,
            "rating": None,
        }
        
        self.assertEqual(result, expected_metadata)

    @patch('mitoolspro.google_utils.youtube.metadata.extract_metadata')
    @patch('mitoolspro.google_utils.youtube.metadata.logger')
    def test_save_metadata_success(self, mock_logger, mock_extract):
        # Setup
        test_metadata = {
            "title": "Test Video",
            "description": "Test description",
            "length": 120,
            "views": 500000,
            "author": "Test Author",
            "channel_id": "UCtest456",
            "channel_url": "https://www.youtube.com/channel/UCtest456",
            "keywords": ["test", "save"],
            "metadata": {"category": "Education"},
            "publish_date": "2023-01-15T10:00:00",
            "rating": 4.2,
        }
        
        mock_extract.return_value = test_metadata
        output_file = Path(self.temp_dir.name) / "metadata.json"
        
        # Execute
        save_metadata(self.test_url, output_file)
        
        # Verify
        mock_extract.assert_called_once_with(self.test_url)
        mock_logger.info.assert_called_once_with("Metadata saved to %s", output_file)

    @patch('mitoolspro.google_utils.youtube.metadata.extract_metadata')
    def test_save_metadata_json_serialization(self, mock_extract):
        # Setup with serializable data only
        test_metadata = {
            "title": "Test Video",
            "publish_date": "2023-01-15T00:00:00",  # Use string instead of datetime
            "length": 120,
        }
        
        mock_extract.return_value = test_metadata
        output_file = Path(self.temp_dir.name) / "metadata.json"
        
        # Execute - this will test the actual file writing
        save_metadata(self.test_url, output_file)
        
        # Verify file was created and contains expected content
        self.assertTrue(output_file.exists())
        
        with output_file.open("r") as f:
            saved_data = json.load(f)
        
        # Check that the data was saved
        self.assertEqual(saved_data["title"], "Test Video")
        self.assertEqual(saved_data["length"], 120)
        self.assertEqual(saved_data["publish_date"], "2023-01-15T00:00:00")


if __name__ == '__main__':
    unittest.main()