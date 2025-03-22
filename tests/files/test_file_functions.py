import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.files.file_functions import (
    build_dir_tree,
    can_move_file_or_folder,
    file_in_folder,
    folder_in_subtree,
    folder_is_subfolder,
    handle_duplicated_filenames,
    remove_characters_from_filename,
    rename_file,
    rename_files_in_folder,
    rename_folders_in_folder,
)


class TestFolderIsSubfolder(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root_folder = Path(self.temp_dir.name)
        self.sub_folder = self.root_folder / "reports"
        self.sub_folder.mkdir()
        self.non_sub_folder = Path(TemporaryDirectory().name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.sub_folder)
        self.assertTrue(result)

    def test_non_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.non_sub_folder)
        self.assertFalse(result)

    def test_same_folder(self):
        result = folder_is_subfolder(self.root_folder, self.root_folder)
        self.assertFalse(result)

    def test_non_existent_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "non_existent_folder"
            subfolder = root / "subfolder"
            result = folder_is_subfolder(root, subfolder)
            self.assertTrue(result)

    def test_string_input(self):
        result = folder_is_subfolder(str(self.root_folder), str(self.sub_folder))
        self.assertTrue(result)


class TestFileInFolder(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_file = self.test_dir / "test_file.txt"
        self.test_file.touch()
        self.subfolder = self.test_dir / "subfolder"
        self.subfolder.mkdir()
        self.subfile = self.subfolder / "subfile.txt"
        self.subfile.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_in_direct_folder(self):
        result = file_in_folder(self.test_dir, self.test_file)
        self.assertTrue(result)

    def test_file_in_subfolder(self):
        result = file_in_folder(self.test_dir, self.subfile)
        self.assertTrue(result)

    def test_file_not_in_folder(self):
        with TemporaryDirectory() as temp_dir:
            outside_file = Path(temp_dir) / "outside_file.txt"
            outside_file.touch()
            result = file_in_folder(self.test_dir, outside_file)
            self.assertFalse(result)

    def test_non_existent_file(self):
        non_existent = self.test_dir / "non_existent.txt"
        result = file_in_folder(self.test_dir, non_existent)
        self.assertFalse(result)

    def test_folder_as_file(self):
        result = file_in_folder(self.test_dir, self.subfolder)
        self.assertFalse(result)

    def test_same_folder(self):
        result = file_in_folder(self.test_dir, self.test_dir)
        self.assertFalse(result)

    def test_invalid_folder_path(self):
        with self.assertRaises(ArgumentValueError):
            file_in_folder(Path("/invalid/path"), self.test_file)

    def test_string_input(self):
        result = file_in_folder(str(self.test_dir), str(self.test_file))
        self.assertTrue(result)


class TestFolderInSubtree(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.branch = (self.root / "reports").resolve(strict=False)
        self.branch.mkdir()
        self.subfolder = self.branch / "2024"
        self.subfolder.mkdir()
        with TemporaryDirectory() as temp_dir:
            self.outside_folder = Path(temp_dir).resolve(strict=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_folder_in_subtree_found(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.branch])
        self.assertEqual(result, self.branch)

    def test_no_folder_in_subtree(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.outside_folder])
        self.assertIsNone(result)

    def test_branch_not_in_root_subtree(self):
        result = folder_in_subtree(
            self.root, self.outside_folder, [self.outside_folder]
        )
        self.assertIsNone(result)

    def test_same_folder_as_root(self):
        result = folder_in_subtree(self.root, self.root, [self.root])
        self.assertIsNone(result)

    def test_string_input(self):
        result = folder_in_subtree(
            str(self.root), str(self.subfolder), [str(self.branch)]
        )
        self.assertEqual(result, self.branch)


class TestCanMoveFileOrFolder(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.source_file = self.test_dir / "source.txt"
        self.source_file.write_text("This is a test file.")
        self.source_folder = self.test_dir / "source_folder"
        self.source_folder.mkdir()
        self.destination_file = self.test_dir / "destination.txt"
        self.destination_folder = self.test_dir / "destination_folder"
        self.destination_folder.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_file_to_file_move(self):
        destination = self.test_dir / "new_name.txt"
        self.assertTrue(can_move_file_or_folder(self.source_file, destination))

    def test_file_to_existing_file_without_overwrite(self):
        self.destination_file.touch()
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(self.source_file, self.destination_file)

    def test_file_to_existing_file_with_overwrite(self):
        self.destination_file.touch()
        self.assertTrue(
            can_move_file_or_folder(
                self.source_file, self.destination_file, overwrite=True
            )
        )

    def test_directory_to_existing_file(self):
        self.assertFalse(
            can_move_file_or_folder(self.source_folder, self.destination_file)
        )

    def test_file_to_directory(self):
        destination = self.destination_folder / "new_file.txt"
        self.assertTrue(can_move_file_or_folder(self.source_file, destination))

    def test_directory_to_directory(self):
        new_directory = self.test_dir / "new_folder"
        new_directory.mkdir()
        self.assertTrue(
            can_move_file_or_folder(self.source_folder, new_directory, overwrite=True)
        )

    def test_source_not_exist(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(non_existent_file, self.destination_file)

    def test_destination_parent_not_exist(self):
        invalid_destination = self.test_dir / "non_existent_folder" / "destination.txt"
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(self.source_file, invalid_destination)

    def test_permission_denied(self):
        self.source_file.chmod(0o000)
        with self.assertRaises(PermissionError):
            can_move_file_or_folder(self.source_file, self.destination_file)
        self.source_file.chmod(0o644)

    def test_path_length_exceeded(self):
        long_name = "a" * 256 + ".txt"
        long_path = self.test_dir / long_name
        with self.assertRaises(OSError):
            can_move_file_or_folder(self.source_file, long_path)


class TestRenameFoldersInFolder(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        (self.test_dir / "folder 1").mkdir()
        (self.test_dir / "folder 2").mkdir()
        (self.test_dir / "already_exists").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_rename(self):
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder_1").exists())
        self.assertTrue((self.test_dir / "folder_2").exists())

    def test_attempt_mode(self):
        rename_folders_in_folder(self.test_dir, attempt=True)
        self.assertTrue((self.test_dir / "folder 1").exists())
        self.assertTrue((self.test_dir / "folder 2").exists())

    def test_custom_char_replacement(self):
        rename_folders_in_folder(
            self.test_dir, char_replacement=lambda name: name.replace(" ", "-")
        )
        self.assertTrue((self.test_dir / "folder-1").exists())
        self.assertTrue((self.test_dir / "folder-2").exists())

    def test_existing_target_folder(self):
        (self.test_dir / "folder_1").mkdir()
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder 1").exists())

    def test_no_change_for_identical_name(self):
        (self.test_dir / "folder_1").mkdir()
        rename_folders_in_folder(
            self.test_dir,
            char_replacement=lambda name: name,
        )
        self.assertTrue((self.test_dir / "folder_1").exists())

    def test_invalid_directory(self):
        with self.assertRaises(ArgumentValueError):
            rename_folders_in_folder(self.test_dir / "non_existent_folder")


class TestRemoveCharactersFromFilename(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_character_removal(self):
        test_file = self.test_dir / "invalid:file?name.txt"
        result = remove_characters_from_filename(test_file)
        self.assertEqual(result.name, "invalidfilename.txt")

    def test_custom_character_removal(self):
        file_path = self.test_dir / "file-name-to-remove-dash.txt"
        expected = "filenametoremovedash.txt"
        result = remove_characters_from_filename(file_path, characters=r"[-]")
        self.assertEqual(result.name, expected)

    def test_filename_without_illegal_characters(self):
        file_path = self.test_dir / "valid_filename.txt"
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, "valid_filename.txt")

    def test_empty_filename(self):
        file_path = self.test_dir / ".txt"
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, ".txt")

    def test_special_unicode_characters(self):
        file_path = self.test_dir / "hello✨world.txt"
        expected = "helloworld.txt"
        result = remove_characters_from_filename(file_path, characters=r"[✨]")
        self.assertEqual(result.name, expected)

    def test_path_object_with_nested_folders(self):
        file_path = self.test_dir / "some/folder/with/invalid:file?name.txt"
        expected = "invalidfilename.txt"
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, expected)

    def test_filename_with_spaces(self):
        file_path = self.test_dir / "file with spaces.txt"
        expected = "filewithspaces.txt"
        result = remove_characters_from_filename(file_path, characters=r"[ ]")
        self.assertEqual(result.name, expected)


class TestHandleDuplicatedFilenames(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.file_path = self.test_dir / "test_file.txt"
        self.file_path.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_conflict(self):
        new_file_path = self.test_dir / "unique_file.txt"
        result = handle_duplicated_filenames(new_file_path)
        self.assertEqual(result, new_file_path)

    def test_single_conflict(self):
        result = handle_duplicated_filenames(self.file_path)
        expected = self.test_dir / "test_file_1.txt"
        self.assertEqual(result, expected)

    def test_multiple_conflicts(self):
        for i in range(1, 3):
            (self.test_dir / f"test_file_{i}.txt").touch()

        result = handle_duplicated_filenames(self.file_path)
        expected = self.test_dir / "test_file_3.txt"
        self.assertEqual(result, expected)

    def test_no_file_extension(self):
        file_without_ext = self.test_dir / "test_file"
        file_without_ext.touch()

        result = handle_duplicated_filenames(file_without_ext)
        expected = self.test_dir / "test_file_1"
        self.assertEqual(result, expected)

    def test_non_existing_directory(self):
        non_existing_file = self.test_dir / "non_existing_folder" / "test_file.txt"
        result = handle_duplicated_filenames(non_existing_file)
        self.assertEqual(result, non_existing_file)


class TestRenameFile(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_file = self.test_dir / "invalid:file?name.txt"
        self.test_file.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_rename_with_sanitization(self):
        rename_file(self.test_file)
        expected_file = self.test_dir / "invalidfilename.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_custom_rename(self):
        new_name = "custom_name.txt"
        rename_file(self.test_file, new_name=new_name)
        expected_file = self.test_dir / new_name
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_conflict_handling(self):
        conflict_file = self.test_dir / "invalidfilename.txt"
        conflict_file.touch()

        rename_file(self.test_file)
        expected_file = self.test_dir / "invalidfilename_1.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_rename_no_extension(self):
        no_ext_file = self.test_dir / "invalid:file?name"
        no_ext_file.touch()

        rename_file(no_ext_file)
        expected_file = self.test_dir / "invalidfilename"
        self.assertTrue(expected_file.exists())
        self.assertFalse(no_ext_file.exists())

    def test_non_existent_file(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaises(ArgumentValueError):
            rename_file(non_existent_file)

    def test_custom_name_with_conflict(self):
        conflict_file = self.test_dir / "custom_name.txt"
        conflict_file.touch()
        rename_file(self.test_file, new_name="custom_name.txt")
        expected_file = self.test_dir / "custom_name_1.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_custom_name_with_conflict_and_overwrite(self):
        conflict_file = self.test_dir / "custom_name.txt"
        conflict_file.touch()
        rename_file(self.test_file, new_name="custom_name.txt", overwrite=True)
        expected_file = self.test_dir / "custom_name.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_callable_new_name(self):
        rename_file(self.test_file, new_name=lambda file: "custom_name.txt")
        expected_file = self.test_dir / "custom_name.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())


class TestRenameFilesInFolder(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        (self.test_dir / "file%&1.txt").touch()
        (self.test_dir / "file%&2.pdf").touch()
        (self.test_dir / "file%&3.TXT").touch()
        (self.test_dir / "non_file").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_rename_all_files(self):
        rename_files_in_folder(self.test_dir)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file1.txt", "file2.pdf", "file3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_file_type_filter(self):
        rename_files_in_folder(self.test_dir, file_types=[".txt"])
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file1.txt", "file3.TXT", "file%&2.pdf"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_custom_function(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "renamed")

        rename_files_in_folder(self.test_dir, renaming_function=custom_renamer)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"renamed1.txt", "renamed2.pdf", "renamed3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_duplicated_name(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "file%&")

        rename_files_in_folder(self.test_dir, renaming_function=custom_renamer)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file%&1_1.txt", "file%&2_1.pdf", "file%&3_1.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_duplicated_name_and_overwrite(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "file%&")

        rename_files_in_folder(
            self.test_dir, renaming_function=custom_renamer, overwrite=True
        )
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file%&1.txt", "file%&2.pdf", "file%&3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_skips_non_files(self):
        rename_files_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "non_file").exists())

    def test_error_handling(self):
        def faulty_renamer(file: str) -> str:
            raise ValueError("Renaming failed")

        try:
            rename_files_in_folder(self.test_dir, renaming_function=faulty_renamer)
        except Exception as e:
            self.fail(f"rename_files_in_folder raised an unexpected exception: {e}")

    def test_rename_with_nonexistent_folder(self):
        with self.assertRaises(ArgumentValueError):
            rename_files_in_folder(self.test_dir / "non_existent_folder")


if __name__ == "__main__":
    unittest.main()
