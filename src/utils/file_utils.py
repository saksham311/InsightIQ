from pathlib import Path


def ensure_directory(directory: Path) -> None:
    """
    Creates the directory if it doesn't exist.
    """
    directory.mkdir(parents=True, exist_ok=True)