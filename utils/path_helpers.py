# /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/utils/path_helpers.py
import os
import sys
from pathlib import Path
from typing import Union, Optional

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.

    This works in both local development and CI environments.
    In GitHub Actions, it respects the GITHUB_WORKSPACE environment variable.
    """
    # Check if running in GitHub Actions
    if "GITHUB_WORKSPACE" in os.environ:
        return Path(os.environ["GITHUB_WORKSPACE"])

    # Otherwise, infer from the current file's location
    # Go up to find the project root (assuming utils is directly under project root)
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_path(relative_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the project root or a specified base directory.

    Args:
        relative_path: A path relative to the project root or base_dir
        base_dir: Optional base directory to resolve from instead of project root

    Returns:
        An absolute Path object
    """
    if base_dir is None:
        base_dir = get_project_root()

    # Handle both string and Path objects
    path = Path(relative_path)

    # If path is already absolute, return it directly
    if path.is_absolute():
        return path

    # Otherwise, resolve it relative to the base directory
    return (base_dir / path).resolve()

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def data_file_path(filename: str) -> Path:
    """
    Get the absolute path to a data file.

    Args:
        filename: Name of the file in the data directory

    Returns:
        Absolute Path to the file
    """
    return resolve_path(f"data/{filename}")

def model_file_path(path: str) -> Path:
    """
    Get the absolute path to a model file or directory.

    Args:
        path: Path relative to the trained_nlu_model directory

    Returns:
        Absolute Path to the model file/directory
    """
    model_dir = resolve_path("trained_nlu_model")
    ensure_dir_exists(model_dir)
    return resolve_path(path, base_dir=model_dir) 