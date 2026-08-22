"""Hugging Face download, extraction, and cache helpers."""

from collections.abc import Sequence
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_CACHE_ROOT = Path("~/.cache/amlgraphx").expanduser()


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset archive cannot be prepared or validated."""


def find_dataset_file(root: Path, filename: str) -> Path:
    """Find a named extracted file below a dataset directory.

    Args:
        root: Extracted dataset directory.
        filename: Expected file name.

    Returns:
        The matching file path.

    Raises:
        FileNotFoundError: If the file is not present.
    """
    direct = root / filename
    if direct.is_file():
        return direct
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Required dataset file not found: {filename} in {root}")


def find_tabular_file(root: Path, preferred_terms: Sequence[str] = ()) -> Path:
    """Find a CSV or Parquet file below an extracted dataset directory.

    Args:
        root: Extracted dataset directory.
        preferred_terms: Terms used to prioritize candidate file names.

    Returns:
        A selected tabular file path.

    Raises:
        FileNotFoundError: If no CSV or Parquet file exists.
    """
    candidates = sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in {".csv", ".parquet"}
            and "__MACOSX" not in path.parts
            and not path.name.startswith("._")
        )
    )
    for term in preferred_terms:
        matches = [path for path in candidates if term.lower() in path.name.lower()]
        if matches:
            return matches[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No CSV or Parquet dataset file found in {root}")


def validate_dataset_files(root: Path, expected_files: Sequence[str]) -> tuple[Path, ...]:
    """Validate and resolve required files in an extracted dataset.

    Args:
        root: Extracted dataset directory.
        expected_files: Required file names.

    Returns:
        Resolved paths in the same order as ``expected_files``.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    return tuple(find_dataset_file(root, filename) for filename in expected_files)


def extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    expected_files: Sequence[str] = (),
) -> Path:
    """Safely extract a ZIP archive and validate its required files.

    Args:
        archive_path: ZIP archive path.
        destination: Directory receiving extracted files.
        expected_files: Optional required extracted file names.

    Returns:
        The extraction directory.

    Raises:
        DatasetDownloadError: If the archive is invalid or unsafe.
        FileNotFoundError: If required files are missing after extraction.
    """
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = (destination / member.filename).resolve()
                if not member_path.is_relative_to(destination):
                    raise DatasetDownloadError(
                        f"Unsafe ZIP member path: {member.filename}"
                    )
            archive.extractall(destination)
    except BadZipFile as exc:
        raise DatasetDownloadError(f"Invalid ZIP archive: {archive_path}") from exc
    if expected_files:
        validate_dataset_files(destination, expected_files)
    return destination


class HuggingFaceDownloader:
    """Download and optionally extract a public Hugging Face dataset file."""

    def __init__(
        self,
        repo_id: str,
        *,
        filename: str | None = None,
        repo_type: str = "dataset",
        revision: str = "main",
        allow_patterns: Sequence[str] | None = None,
        cache_dir: Path = DEFAULT_CACHE_ROOT,
        local_dir: Path | None = None,
    ) -> None:
        """Configure a Hugging Face download.

        Args:
            repo_id: Hugging Face repository identifier.
            filename: Optional archive/file to download from the repository.
            repo_type: Hugging Face repository type.
            revision: Repository branch, tag, or commit.
            allow_patterns: File patterns for snapshot downloads.
            cache_dir: Local Hugging Face cache directory.
            local_dir: Optional local download directory.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.repo_type = repo_type
        self.revision = revision
        self.allow_patterns = allow_patterns
        self.cache_dir = Path(cache_dir).expanduser()
        self.local_dir = Path(local_dir).expanduser() if local_dir else None

    def download(
        self,
        *,
        target_dir: Path | None = None,
        expected_files: Sequence[str] = (),
    ) -> Path:
        """Download the configured file or snapshot and return its local root.

        When ``filename`` is a ZIP archive, it is extracted into ``target_dir``
        and the extracted files are validated before returning.

        Args:
            target_dir: Optional extraction or output directory.
            expected_files: Required files after archive extraction.

        Returns:
            Local snapshot or extracted dataset directory.
        """
        if self.filename is None:
            downloaded = snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                allow_patterns=self.allow_patterns,
                cache_dir=self.cache_dir,
                local_dir=self.local_dir,
            )
            return Path(downloaded).expanduser().resolve()

        kwargs: dict[str, object] = {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
        }
        if self.local_dir is not None:
            kwargs["local_dir"] = self.local_dir
        archive_path = Path(hf_hub_download(**kwargs)).expanduser().resolve()
        destination = target_dir or self.local_dir or (
            self.cache_dir / self.repo_id.replace("/", "--") / Path(self.filename).stem
        )
        return extract_zip(archive_path, Path(destination), expected_files=expected_files)
