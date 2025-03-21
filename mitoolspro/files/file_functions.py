import os
import shutil
from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Union

from treelib import Tree

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.utils.functions import remove_characters_from_string


def build_dir_tree(
    directory: PathLike, tree: Optional[Tree] = None, parent: Optional[PathLike] = None
) -> Tree:
    if tree is None:
        tree = Tree()
        tree.create_node(directory.name, str(directory))
        parent = str(directory)
    for item in sorted(directory.iterdir()):
        node_id = str(item)
        if item.is_dir():
            tree.create_node(item.name, node_id, parent=parent)
            build_dir_tree(item, tree, parent=node_id)
        else:
            tree.create_node(item.name, node_id, parent=parent)
    return tree


def folder_is_subfolder(root_folder: PathLike, folder_to_check: PathLike) -> bool:
    root_folder = Path(root_folder)
    folder_to_check = Path(folder_to_check)
    try:
        root_folder = root_folder.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(
            f"Invalid 'root_folder'={root_folder} path provided: {e}"
        )
    try:
        folder_to_check = folder_to_check.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(
            f"Invalid 'folder_to_check'={folder_to_check} path provided: {e}"
        )
    if root_folder == folder_to_check:
        return False
    return root_folder in folder_to_check.parents


def file_in_folder(folder: PathLike, file_to_check: PathLike) -> bool:
    folder = Path(folder)
    file_to_check = Path(file_to_check)
    try:
        folder = folder.resolve(strict=True)
    except Exception as e:
        raise ArgumentValueError(f"Invalid 'folder'={folder} path provided: {e}")
    try:
        file_to_check = file_to_check.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(
            f"Invalid 'file_to_check'={file_to_check} path provided: {e}"
        )
    if not file_to_check.is_file():
        return False
    return folder in file_to_check.parents or folder == file_to_check.parent


def folder_in_subtree(
    root_folder: PathLike, branch_folder: PathLike, folders_to_check: List[PathLike]
) -> Union[Path, None]:
    root_folder = Path(root_folder).resolve(strict=False)
    branch_folder = Path(branch_folder).resolve(strict=False)
    folders_to_check = {
        Path(folder).resolve(strict=False) for folder in folders_to_check
    }
    if not folder_is_subfolder(root_folder, branch_folder):
        return None
    for folder in branch_folder.parents:
        if folder in folders_to_check:
            return folder
        if folder == root_folder:
            break
    return None


def can_move_file_or_folder(
    source: PathLike, destination: PathLike, overwrite: bool = False
) -> bool:
    try:
        source = Path(source).resolve(strict=True)
    except FileNotFoundError as e:
        raise ArgumentValueError(f"Invalid 'source'={source} provided.")
    destination = Path(destination).resolve(strict=False)
    if not destination.parent.exists():
        raise ArgumentValueError(
            f"'destination.parent={destination.parent}' does not exist."
        )
    if destination.exists() and not overwrite:
        raise ArgumentValueError(f"'{destination}' already exists.")
    if not os.access(source, os.R_OK):
        raise PermissionError(f"Read permission denied for '{source}'.")
    if not os.access(destination.parent, os.W_OK):
        raise PermissionError(f"Write permission denied for '{destination.parent}'.")
    if len(str(destination.name)) > 255:
        raise OSError(
            f"The path 'destination.name={destination.name}' exceeds the maximum length allowed."
        )
    if source.stat().st_dev != destination.parent.stat().st_dev:
        src_size = source.stat().st_size
        free_space = destination.parent.stat().st_blocks * 512  # Approximate free space
        if free_space < src_size:
            raise OSError("Insufficient space on the destination drive.")
    if source == destination:
        return False
    if source.is_dir() and destination.is_file():
        return False
    if source.is_file() and destination.parent.is_dir():
        return True
    if source.is_dir() and destination.is_dir():
        return True
    return False


def rename_folders_in_folder(
    folder_path: PathLike,
    char_replacement: Callable[[str], str] = None,
    attempt: bool = False,
    overwrite: bool = False,
) -> None:
    folder_path = Path(folder_path).resolve(strict=False)
    if not folder_path.is_dir():
        raise ArgumentValueError(f"{folder_path} is not a valid directory.")
    char_replacement = char_replacement or (lambda name: name.replace(" ", "_"))
    for folder in folder_path.iterdir():
        if folder.is_dir():
            new_name = char_replacement(folder.name)
            new_path = folder_path / new_name
            if folder == new_path:
                continue
            if new_path.exists() and not overwrite:
                print(
                    f"Skipping '{folder.name}' → '{new_name}' (target already exists)"
                )
                continue
            if attempt:
                print(
                    f"[Attempt] Renaming '{folder.name}' to '{new_name}' results in {can_move_file_or_folder(folder, new_path)}"
                )

            else:
                print(f"Renaming '{folder.name}' to '{new_name}'")
                shutil.move(str(folder), str(new_path))


def remove_characters_from_filename(
    file_path: PathLike, characters: str = None
) -> Path:
    file_path = Path(file_path)
    filename = remove_characters_from_string(
        string=file_path.stem, characters=characters
    )
    return file_path.with_name(f"{filename}{file_path.suffix}")


def handle_duplicated_filenames(file_path: Path) -> Path:
    counter = 1
    new_file = file_path
    while new_file.exists():
        new_file = file_path.with_name(f"{file_path.stem}_{counter}{file_path.suffix}")
        counter += 1
    return new_file


def rename_file(
    file: PathLike,
    new_name: Union[PathLike, str, Callable[[PathLike], str]] = None,
    attempt: bool = False,
    overwrite: bool = False,
) -> None:
    file = Path(file)
    if callable(new_name):
        new_name_str = new_name(file.name)
    elif isinstance(new_name, str) or new_name is None:
        new_name_str = new_name
    else:
        new_name_str = new_name.name
    sanitized_name = (
        remove_characters_from_filename(file)
        if new_name_str is None
        else file.with_name(new_name_str)
    )
    new_file = (
        handle_duplicated_filenames(sanitized_name) if not overwrite else sanitized_name
    )
    if attempt:
        print(
            f"[Attempt] Renaming '{file}' to '{new_name_str}' results in {can_move_file_or_folder(file, new_file, overwrite=overwrite)}"
        )
    elif can_move_file_or_folder(file, new_file, overwrite=overwrite):
        shutil.move(str(file), str(new_file))
        print(f"Renamed '{file.name}' to '{new_file.name}'")


def rename_files_in_folder(
    folder_path: PathLike,
    file_types: List[str] = None,
    renaming_function: Callable[[str], str] = None,
    attempt: bool = False,
    overwrite: bool = False,
) -> None:
    try:
        folder = Path(folder_path).resolve(strict=True)
    except FileNotFoundError as e:
        raise ArgumentValueError(f"Invalid 'folder_path'={folder_path} provided.")
    for file in folder.iterdir():
        if not file.is_file():
            continue  # Skip non-files
        if file_types and file.suffix.lower() not in file_types:
            continue  # Skip files not in the specified types
        try:
            if renaming_function is not None:
                new_name = renaming_function(file.name)
            rename_file(
                file=file,
                new_name=None if renaming_function is None else new_name,
                attempt=attempt,
                overwrite=overwrite,
            )
        except Exception as e:
            print(
                f"Error processing '{file.name}' with 'renaming_function'={renaming_function}: {e}"
            )
