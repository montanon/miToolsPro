import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import sys
from io import StringIO

from mitoolspro.cli import init_project, main


class TestInitProject(unittest.TestCase):
    @patch('mitoolspro.cli.Project')
    @patch('mitoolspro.cli.logger')
    def test_init_project_basic(self, mock_logger, mock_project_class):
        # Create mock args
        args = argparse.Namespace(
            name='test_project',
            root=Path('/tmp/test'),
            version='v1.0'
        )
        
        # Call the function
        init_project(args)
        
        # Verify Project was instantiated with correct arguments
        mock_project_class.assert_called_once_with(
            project_name='test_project',
            root=Path('/tmp/test'),
            version='v1.0'
        )
        
        # Verify logger was called
        mock_logger.info.assert_called_once_with(
            "Initialized project '%s' in %s", 
            'test_project', 
            Path('/tmp/test')
        )

    @patch('mitoolspro.cli.Project')
    @patch('mitoolspro.cli.logger')
    def test_init_project_with_default_values(self, mock_logger, mock_project_class):
        # Create args with default values
        args = argparse.Namespace(
            name='my_project',
            root=Path.cwd(),
            version='v0'
        )
        
        init_project(args)
        
        mock_project_class.assert_called_once_with(
            project_name='my_project',
            root=Path.cwd(),
            version='v0'
        )
        
        mock_logger.info.assert_called_once_with(
            "Initialized project '%s' in %s", 
            'my_project', 
            Path.cwd()
        )


class TestMain(unittest.TestCase):
    def setUp(self):
        # Store original sys.argv
        self.original_argv = sys.argv[:]
    
    def tearDown(self):
        # Restore original sys.argv
        sys.argv = self.original_argv

    @patch('mitoolspro.cli.init_project')
    def test_main_init_command_basic(self, mock_init_project):
        # Mock command line arguments
        sys.argv = ['cli.py', 'init', 'test_project']
        
        main()
        
        # Verify init_project was called
        mock_init_project.assert_called_once()
        args = mock_init_project.call_args[0][0]
        self.assertEqual(args.name, 'test_project')
        self.assertEqual(args.root, Path.cwd())
        self.assertEqual(args.version, 'v0')

    @patch('mitoolspro.cli.init_project')
    def test_main_init_command_with_options(self, mock_init_project):
        sys.argv = [
            'cli.py', 'init', 'my_project', 
            '--root', '/custom/path', 
            '--version', 'v2.1'
        ]
        
        main()
        
        mock_init_project.assert_called_once()
        args = mock_init_project.call_args[0][0]
        self.assertEqual(args.name, 'my_project')
        self.assertEqual(args.root, Path('/custom/path'))
        self.assertEqual(args.version, 'v2.1')

    @patch('sys.stdout', new_callable=StringIO)
    def test_main_no_command_shows_help(self, mock_stdout):
        sys.argv = ['cli.py']
        
        main()
        
        # Check that help was printed
        output = mock_stdout.getvalue()
        self.assertIn('miToolsPro CLI', output)
        self.assertIn('Available commands', output)
        self.assertIn('init', output)

    @patch('sys.stderr', new_callable=StringIO)
    def test_main_unknown_command_shows_error(self, mock_stderr):
        sys.argv = ['cli.py', 'unknown_command']
        
        with self.assertRaises(SystemExit) as cm:
            main()
        
        # Check that exit status is 2 (argument error)
        self.assertEqual(cm.exception.code, 2)
        
        # Check that error message was printed to stderr
        error_output = mock_stderr.getvalue()
        self.assertIn('invalid choice', error_output)
        self.assertIn('unknown_command', error_output)

    def test_argument_parser_init_subcommand(self):
        # Test the argument parser directly by simulating main() logic
        parser = argparse.ArgumentParser(description="miToolsPro CLI")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        init_parser = subparsers.add_parser("init", help="Initialize a new project")
        init_parser.add_argument("name", help="Name of the project")
        init_parser.add_argument(
            "--root",
            type=Path,
            default=Path.cwd(),
            help="Root directory for the project (default: current directory)",
        )
        init_parser.add_argument(
            "--version",
            default="v0",
            help="Initial version of the project (default: v0)",
        )

        # Test parsing valid init command
        args = parser.parse_args(['init', 'test_project'])
        self.assertEqual(args.command, 'init')
        self.assertEqual(args.name, 'test_project')
        self.assertEqual(args.root, Path.cwd())
        self.assertEqual(args.version, 'v0')

        # Test parsing init command with all options
        args = parser.parse_args([
            'init', 'my_project', 
            '--root', '/tmp/projects', 
            '--version', 'v1.5'
        ])
        self.assertEqual(args.command, 'init')
        self.assertEqual(args.name, 'my_project')
        self.assertEqual(args.root, Path('/tmp/projects'))
        self.assertEqual(args.version, 'v1.5')

    def test_path_argument_conversion(self):
        # Test that --root argument properly converts string to Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=Path, default=Path.cwd())
        
        args = parser.parse_args(['--root', '/test/path'])
        self.assertIsInstance(args.root, Path)
        self.assertEqual(args.root, Path('/test/path'))
        
        # Test default value
        args = parser.parse_args([])
        self.assertIsInstance(args.root, Path)
        self.assertEqual(args.root, Path.cwd())


class TestCLIIntegration(unittest.TestCase):
    """Integration tests that test the CLI functionality end-to-end"""
    
    def setUp(self):
        self.original_argv = sys.argv[:]
    
    def tearDown(self):
        sys.argv = self.original_argv

    @patch('mitoolspro.cli.Project')
    @patch('mitoolspro.cli.logger')
    def test_full_init_workflow(self, mock_logger, mock_project_class):
        """Test the complete workflow from command line to project creation"""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock sys.argv for init command
            sys.argv = [
                'cli.py', 'init', 'integration_test_project',
                '--root', str(temp_path),
                '--version', 'v0.1.0'
            ]
            
            # Run main
            main()
            
            # Verify Project was created with correct arguments
            mock_project_class.assert_called_once_with(
                project_name='integration_test_project',
                root=temp_path,
                version='v0.1.0'
            )
            
            # Verify logging
            mock_logger.info.assert_called_once_with(
                "Initialized project '%s' in %s",
                'integration_test_project',
                temp_path
            )

    @patch('mitoolspro.cli.Project')
    @patch('mitoolspro.cli.logger')
    def test_init_with_relative_path(self, mock_logger, mock_project_class):
        """Test init command with relative path"""
        sys.argv = [
            'cli.py', 'init', 'relative_project',
            '--root', './projects/new',
            '--version', 'v1.0'
        ]
        
        main()
        
        expected_path = Path('./projects/new')
        mock_project_class.assert_called_once_with(
            project_name='relative_project',
            root=expected_path,
            version='v1.0'
        )


if __name__ == '__main__':
    unittest.main()